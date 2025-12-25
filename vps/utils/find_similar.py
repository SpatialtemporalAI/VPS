import argparse
import collections.abc as collections
from pathlib import Path
import re
from typing import Optional, List, Dict, Union
import h5py
import numpy as np
import torch
import json
import time
from hloc import logger
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_image_lists
from hloc.utils.read_write_model import read_images_binary
import logging
from sklearn.cluster import KMeans

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.ndarray,
    num_select: int,
    min_score: Optional[float] = None,
):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()
    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def spatial_filter(last_pose, ref_poses_tensor, spatial_radius, device):
    """
    db_names: list[str]
    db_desc: torch.Tensor [N, D]
    last_pose: np.ndarray or torch.Tensor [3,]
    ref_poses_tensor: torch.Tensor [N, 3]
    spatial_radius: float
    device: str
    return: numpy.ndarray [N,]
    """
    start = time.time()
    last_pose = torch.from_numpy(last_pose).float().to(device)
    ref_poses_tensor = ref_poses_tensor.to(device)
    ref_poses_tensor = ref_poses_tensor[:3, 3]
    # 计算欧氏距离
    dists = torch.norm(ref_poses_tensor - last_pose, dim=1)  # [N,]
    invalid_tensor = dists > spatial_radius
    valid_indices = torch.where(~invalid_tensor)[0]
    invalid = invalid_tensor.cpu().numpy()
    logging.info(f"空间过滤后剩余: {len(valid_indices)} 张参考图像")
    end = time.time()
    logging.info(f"空间过滤时间: {end - start} 秒")
    return invalid


def find_similar(
    query_descriptors,
    db_descriptors,
    db_names,
    db_desc,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    similarity_threshold=0.7,
    last_pose=None,
    spatial_radius=None,
    use_spatial_filtering=False,
    ref_poses_tensor=None,
):

    query_names_h5 = list_h5_names(query_descriptors)
    if len(db_names) == 0:
        logging.error("Could not find any database image.")
        raise ValueError("Could not find any database image.")
    query_names = parse_names(query_prefix, query_list, query_names_h5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_desc = get_descriptors(query_names, query_descriptors)
    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    # if use_spatial_filtering \
    # and last_pose is not None and len(last_pose) > 0 \
    # and spatial_radius is not None \
    # and ref_poses_tensor is not None:
    #     invalid_tensor = spatial_filter(last_pose, ref_poses_tensor, spatial_radius, device)
    #     self = invalid_tensor | self
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=similarity_threshold)  
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))



def find_similar_vpr_pose(
    query_name,        #str query name
    query_descriptors, #query 特征   Path
    db_descriptors,    #refs 特征路径  list[path]
    db_names,          #refs 名称 list
    db_desc,           # db map {name: descriptor}  torch.Tensor [N, D]
    output,
    num_matched,
    size_num_matched, #几倍num_matched查找
    ref_poses_tensor,
    query_prefix=None,
    query_list=None,
    # only a couple of knobs:
    outlier_percentile=95,   # remove far-away poses among top-2k by this percentile
    ang_weight=0.6,          # weight for angular term (radians)
    pos_weight=0.2,          # weight for position term (meters)
    sim_weight=0.2,          # weight to incorporate VPR similarity into final score (0..1)
    verbose=False
):
    """
    目的:纯vpr的相似度排序 图像可能过于类似,相机位置差异太小 不利于后续优化,尤其是k不是太大的时候
        小k 找到更多更合理 分布更散的相机  让k=5 达到k = 10的效果????
    Implements: top-2k by VPR -> seed (top1) -> remove pos outliers (percentile) ->
                greedy pick by maximizing min(weighted pose distance) combined with sim.
    Returns: selected_db_indices (list), selected_db_names (list), info dict
    """

    # top-N prefetch (N = min(2*k, M))
    k = num_matched
    N = int(size_num_matched*num_matched)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_desc = get_descriptors([query_name], query_descriptors)
    # Avoid self-matching
    self = np.array([query_name])[:, None] == np.array(db_names)[None]
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    pairs = pairs_from_score_matrix(sim, self, N)
    topN_idx = [int(j) for i,j in pairs]
    topN_names = [db_names[i] for i in topN_idx]
    topN_poses_tensor = ref_poses_tensor[topN_idx]
    topN_poses = topN_poses_tensor.cpu().numpy()
    topN_scores_tensor = sim[0][topN_idx]
    topN_scores = topN_scores_tensor.cpu().numpy()
    # print(f"topN_names: {topN_names}")
    # print(f"topN_idx: {topN_idx}")
    # print(f"topN_scores: {topN_scores}")
    # print(f"topN_poses: {topN_poses}")


    # for i in range(len(topN_scores)):
    #     if topN_scores[i] < topN_scores[0] * 0.6:
    #         N = i
    #         print(f"N: {N}")
    #         break
    # if N < k:
    #     N = k
    # pairs = pairs_from_score_matrix(sim, self, N)
    # topN_idx = [int(j) for i,j in pairs]
    # topN_names = [db_names[i] for i in topN_idx]
    # topN_poses_tensor = ref_poses_tensor[topN_idx]
    # topN_poses = topN_poses_tensor.cpu().numpy()
    # topN_scores_tensor = sim[0][topN_idx]
    # topN_scores = topN_scores_tensor.cpu().numpy()









    centers_all = topN_poses[:, :3, 3]
    R_all = topN_poses[:, :3, :3]
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forwards_all = (R_all @ z.reshape(3,1)).squeeze(-1)
    f_norm = np.linalg.norm(forwards_all, axis=1, keepdims=True) + 1e-12
    forwards_all = forwards_all / f_norm
    sims_all = topN_scores

    # --- 2. 离群点剔除 (Outlier Removal) ---
    # 目的：根据xyz pose 过滤明显的 VPR 匹配错误

    median_pos = np.median(centers_all, axis=0)
    dists_to_med = np.linalg.norm(centers_all - median_pos, axis=1)
    cutoff = np.percentile(dists_to_med, outlier_percentile)
    # 始终保留 Seed Top-1 (最相似的)，即使它可能偏离中位数（防止误杀）
    keep_mask = (dists_to_med <= cutoff)
    keep_mask[0] = True 
    valid_local_indices = np.where(keep_mask)[0]

    # 重新切片数据，只保留 valid candidates
    cand_indices = valid_local_indices # 指向 topN_poses 的索引
    cand_centers = centers_all[cand_indices]
    cand_forwards = forwards_all[cand_indices]
    cand_scores = sims_all[cand_indices]
    # 归一化分数
    s_min, s_max = cand_scores.min(), cand_scores.max()
    cand_norm_scores = (cand_scores - s_min) / (s_max - s_min + 1e-6)

    cand_names = [db_names[topN_idx[i]] for i in cand_indices]
    # print(f"cand_names: {cand_names}")
    # print(f"cand_indices: {cand_indices}")
    # print(f"cand_scores: {cand_scores}")
    # --- 3. 贪心选择 (Greedy Selection for Diversity) ---
    # 目标：选择 k 个，最大化 "min_dist_to_selected"
    # 初始化：选择seed vpr分最高的
    best_first_idx = np.argmax(cand_scores)
    
    selected_local_indices = [cand_indices[best_first_idx]] # 存的是 topN 列表里的下标
    selected_mask = np.zeros(len(cand_indices), dtype=bool)
    selected_mask[best_first_idx] = True
    
    # 归一化因子预计算
    # 计算候选者包围盒大小，用于归一化距离，防止 pos_weight 难以调整
    if len(cand_centers) > 1:
        scene_scale = np.max(np.linalg.norm(cand_centers - np.mean(cand_centers, axis=0), axis=1)) * 2.0
        scene_scale = max(scene_scale, 1.0) # 避免除零
    else:
        scene_scale = 1.0

    def get_pose_distance(idx_a_local, idx_b_local):
        # 输入是 cand_indices 的数组下标
        pos_a, fwd_a = cand_centers[idx_a_local], cand_forwards[idx_a_local]
        pos_b, fwd_b = cand_centers[idx_b_local], cand_forwards[idx_b_local]
        
        # 1. 位置距离 (Normalized)
        dist_pos = np.linalg.norm(pos_a - pos_b) / scene_scale
        
        # 2. 角度距离 (Normalized 0~1)
        # dot product clip to -1..1
        dot = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
        dist_ang = np.arccos(dot) / np.pi
        
        return pos_weight * dist_pos + ang_weight * dist_ang
    # 迭代选择直到满 k 个
    while len(selected_local_indices) < k and len(selected_local_indices) < len(cand_indices):
        best_candidate_idx = -1
        best_score = -1e9
        
        # 遍历所有未选中的候选者
        for i in range(len(cand_indices)):
            if selected_mask[i]:
                continue
            
            # Farthest Point Sampling 逻辑:
            # 计算当前候选者 i 到 "已选集合" 中最近点的距离
            # 我们希望这个 "最近距离" 越大越好 (离已选的越远越好)
            min_dist_to_current_set = 1e9
            dists = []
            # 获取已选点的 indices (在 cand 数组中的下标)
            curr_sel_indices = np.where(selected_mask)[0]
            for sel_i in curr_sel_indices:
                d = get_pose_distance(i, sel_i)
                dists.append(d)
            min_dist = min(dists)
            
            # 综合打分: 
            # 距离分 (Geometry Diversity) + 相似度分 (Visual Reliability)
            # cand_norm_scores[i] 大小归一化了
            # 这里的 sim_weight 是为了打破纯几何的 tie，或者避免选到 VPR 分数太低的点
            score = min_dist + sim_weight * float(cand_norm_scores[i])
            
            if score > best_score:
                best_score = score
                best_candidate_idx = i
        
        if best_candidate_idx != -1:
            selected_mask[best_candidate_idx] = True
            # 记录原始 TopN 中的索引，方便最后映射回全局
            real_topn_idx = cand_indices[best_candidate_idx]
            selected_local_indices.append(real_topn_idx)
        else:
            break
    # --- 4. 最终输出 ---
    selected_idx = [topN_idx[i] for i in selected_local_indices]
    selected_names = [db_names[i] for i in selected_idx]
    info = {
        "topN_idx": topN_idx,
        "kept_after_outlier": len(cand_indices),
        "selected_globals": selected_idx
    }
    pairs = [(query_name, db_names[j]) for j in selected_idx]
    # print(pairs)
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    return selected_idx, selected_names, info

def find_similar_vpr_pose_kmeans(
    query_name,        # str query name
    query_descriptors, # query 特征 Path
    db_descriptors,    # refs 特征路径 list[path]
    db_names,          # refs 名称 list
    db_desc,           # db map {name: descriptor} torch.Tensor [N, D]
    output,
    num_matched,       # Target k
    ref_poses_tensor,
    query_prefix=None,
    query_list=None,
    # knobs:
    outlier_percentile=95,   # 离群值剔除阈值
    verbose=False
):
    """
    [SOTA Strategy] K-Means Grouping Selection
    Implements: Top-N VPR -> Outlier Removal -> Spatial Clustering (K-Means) -> Cluster-wise Best Score Selection.
    
    该方法通过将候选者在空间上聚类为 k 组，并在每组中选择 VPR 分数最高的图像，
    从而在保证几何多样性（覆盖不同位置）的同时，最大化视觉匹配的可靠性。
    """
    print("kmeans")
    # --- 1. Top-N Prefetch ---
    k = num_matched
    # 预取倍数：建议 N >= 3*k 到 5*k，给聚类留出足够的样本空间
    N = 4*k
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_desc = get_descriptors([query_name], query_descriptors)
    
    # VPR Similarity Calculation
    # Avoid self-matching logic (assuming db_names checks)
    # ... (Keep your original logic here)
    query_desc = get_descriptors([query_name], query_descriptors)
    # Avoid self-matching
    self = np.array([query_name])[:, None] == np.array(db_names)[None]
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    pairs = pairs_from_score_matrix(sim, self, N)
    topN_idx = [int(j) for i,j in pairs]
    topN_names = [db_names[i] for i in topN_idx]
    topN_poses_tensor = ref_poses_tensor[topN_idx]
    topN_poses = topN_poses_tensor.cpu().numpy()
    topN_scores_tensor = sim[0][topN_idx]
    topN_scores = topN_scores_tensor.cpu().numpy()
    

    

    # Extract Centers (XYZ)
    centers_all = topN_poses[:, :3, 3]
    sims_all = topN_scores

    # --- 2. 离群点剔除 (Outlier Removal) ---
    # 目的：根据物理距离剔除明显的 VPR 匹配错误（如重复纹理导致的远距离瞬移）
    if len(centers_all) > 2:
        median_pos = np.median(centers_all, axis=0)
        dists_to_med = np.linalg.norm(centers_all - median_pos, axis=1)
        cutoff = np.percentile(dists_to_med, outlier_percentile)
        
        # 始终保留 Top-1 (VPR最相似的)，即使它偏离中位数，防止误杀真值
        keep_mask = (dists_to_med <= cutoff)
        keep_mask[0] = True 
        valid_local_indices = np.where(keep_mask)[0]
    else:
        valid_local_indices = np.arange(len(centers_all))

    # 准备聚类数据
    # cand_indices 是指向 topN 数组的下标
    cand_indices = valid_local_indices 
    cand_centers = centers_all[cand_indices]
    cand_scores = sims_all[cand_indices]

    if verbose:
        print(f"[K-Means] Top-N: {len(centers_all)}, After Outlier Removal: {len(cand_centers)}")

    # --- 3. K-Means 聚类选择 (Cluster & Pick Best) ---
    selected_local_indices = [] # 存储选中的 cand_indices 中的值 (即 topN 的下标)

    # 如果有效候选者不足 k 个，直接全选
    if len(cand_centers) <= k:
        selected_local_indices = cand_indices.tolist()
    else:
        # 3.1 运行 K-Means
        # n_clusters = k
        # n_init=10: 运行10次取最优，保证稳定性
        try:
            print("kmeans fit")
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(cand_centers)
            print("dawdawd")
            # 3.2 在每个 Cluster 中选择 Score 最高的
            for cluster_id in range(k):
                # 找到属于该簇的所有点
                cluster_mask = (labels == cluster_id)
                
                if not np.any(cluster_mask):
                    continue
                
                # 获取该簇成员在 cand_centers 中的下标
                member_indices_in_cand = np.where(cluster_mask)[0]
                
                # 获取这些成员的 VPR Scores
                member_scores = cand_scores[member_indices_in_cand]
                
                # 找到簇内分数最高的那个点的下标
                best_in_cluster_loc = np.argmax(member_scores)
                best_idx_in_cand = member_indices_in_cand[best_in_cluster_loc]
                
                # 记录对应的 topN 下标
                selected_local_indices.append(cand_indices[best_idx_in_cand])
                
        except Exception as e:
            if verbose: print(f"[K-Means Error] Fallback to Top-K. Reason: {e}")
            # Fallback: 直接选分数最高的 k 个
            top_k_local = np.argsort(-cand_scores)[:k]
            selected_local_indices = [cand_indices[i] for i in top_k_local]

    # --- 4. 补齐逻辑 (Gap Filling) ---
    # 极其罕见的情况：K-Means 产生的簇少于 k (空簇) 或者 初始候选不足
    # 策略：从剩余未选中的候选者中，按 VPR 分数从高到低补齐
    if len(selected_local_indices) < k:
        current_set = set(selected_local_indices)
        # 对剩余所有点按分数排序
        remaining_sorted = sorted(
            cand_indices, 
            key=lambda idx: -sims_all[idx] # 使用全局 scores 数组
        )
        
        for idx in remaining_sorted:
            if len(selected_local_indices) >= k:
                break
            if idx not in current_set:
                selected_local_indices.append(idx)
                current_set.add(idx)

    # --- 5. 最终输出转换 ---
    # 将 topN 下标转换为全局 DB 下标
    selected_globals = [topN_idx[i] for i in selected_local_indices]
    
    # 按照 VPR 分数重新降序排列 (可选，通常下游任务喜欢有序的)
    # 获取选定项的 scores
    final_scores = [sims_all[i] for i in selected_local_indices]
    #以此排序
    sorted_pairs = sorted(zip(selected_globals, final_scores), key=lambda x: -x[1])
    selected_globals = [p[0] for p in sorted_pairs]
    
    selected_names = [db_names[i] for i in selected_globals]

    info = {
        "topN_idx": topN_idx,
        "kept_after_outlier": len(cand_centers),
        "selected_globals": selected_globals,
        "method": "kmeans_clustering"
    }

    pairs = [(query_name, name) for name in selected_names]
    if output:
        with open(output, "w") as f:
            f.write("\n".join(" ".join([i, j]) for i, j in pairs))
            
    return selected_globals, selected_names, info


# def find_similar_vpr_pose_v5_maximized(
#     query_name,
#     query_descriptors, # ... 其他参数 ...
#     db_desc,           
#     db_names,
#     output,
#     num_matched,          # K=10
#     ref_poses_tensor,
#     # --- 关键参数 ---
#     prefetch_size=100,    # 1. 暴力拉取：直接看前100个，足够大了
#     similarity_keep_ratio=0.85, # 2. 界限：只要分数 > Top1 * 0.85，就算“相似池子”里的
#     # --- 多样性参数 ---
#     pos_weight=0.3,
#     ang_weight=0.5,
#     sim_weight=0.2,
#     outlier_percentile=95
# ):
#     """
#     V5 极简最大化版：
#     1. 先拿 Top-100。
#     2. 用 Top-1 分数划线，保留所有高分样本，最大化候选池。
#     3. 在这个大池子里贪心选最散的 10 个。
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     k = num_matched
    
#     # --- 第一步：暴力拉取 Top-100 ---
#     # 不要一点点 loop，直接算！
#     query_desc = get_descriptors([query_name], query_descriptors)
#     sim_matrix = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    
#     # 获取 Top-100 的索引和分数
#     # topk 直接返回排好序的结果，速度极快
#     top_scores, top_indices = torch.topk(sim_matrix, k=min(prefetch_size, sim_matrix.shape[1]), dim=1)
    
#     # 转 numpy
#     pool_indices = top_indices[0].cpu().numpy()
#     pool_scores = top_scores[0].cpu().numpy()
    
#     # --- 第二步：最大化候选池 (划定相似界限) ---
#     # 核心逻辑：谁是“不相似”的？ 只有那些分数掉得太狠的才是不相似。
#     # 我们以 Top-1 为基准。如果 Top-1 是 0.9，那么 0.9*0.85 = 0.765 以上的都算相似。
#     # 如果 Top-1 只有 0.6，那么 0.6*0.85 = 0.51 以上的都算相似。
#     # 这就实现了“自适应最大化”。
    
#     top1_score = pool_scores[0]
#     threshold = top1_score * similarity_keep_ratio
    
#     # 找到截断点：所有大于阈值的都保留
#     # np.argmax 在布尔数组中会返回第一个 False 的位置，如果没有 False 返回 0
#     # 我们找第一个 < threshold 的位置
#     mask = pool_scores >= threshold
#     valid_count = np.sum(mask)
    
#     # 哪怕池子很大，我们至少也要保留 k 个，防止阈值切太狠
#     valid_count = max(valid_count, k) 
    
#     # 最终的候选池 (Candidate Pool)
#     # 这就是你要的“最大化的相似序列”
#     cand_indices = pool_indices[:valid_count]
#     cand_scores = pool_scores[:valid_count]
    
#     # 对应的位姿
#     cand_poses = ref_poses_tensor[cand_indices].cpu().numpy()
#     cand_centers = cand_poses[:, :3, 3]

#     print(f"DEBUG: Top1分数={top1_score:.4f}, 阈值={threshold:.4f}, 最大化池子大小={len(cand_indices)}")

#     # --- 第三步：贪心多样性选择 (逻辑不变) ---
#     # 1. 局部尺度计算
#     median_pos = np.median(cand_centers, axis=0)
#     dists = np.linalg.norm(cand_centers - median_pos, axis=1)
#     scene_scale = max(np.median(dists) * 2.0, 1.0)
#     outlier_cut = np.percentile(dists, outlier_percentile)

#     # 2. 准备方向向量
#     R_pool = cand_poses[:, :3, :3]
#     z_ref = np.array([0, 0, 1.0])
#     fwds = (R_pool @ z_ref.reshape(3,1)).squeeze(-1)
#     fwds /= (np.linalg.norm(fwds, axis=1, keepdims=True) + 1e-12)
    
#     # 3. 归一化分数 (0~1)
#     s_min, s_max = cand_scores.min(), cand_scores.max()
#     norm_scores = (cand_scores - s_min) / (s_max - s_min + 1e-6)

#     # 4. 开始选
#     selected_local = [0] # 必选 Top1
    
#     while len(selected_local) < k:
#         best_score = -1e9
#         best_idx = -1
        
#         for i in range(len(cand_indices)):
#             if i in selected_local: continue
#             if dists[i] > outlier_cut: continue # 离群点太远不要
            
#             # 找离已选集最远的点
#             min_dist = 1e9
#             for sel in selected_local:
#                 d_pos = np.linalg.norm(cand_centers[i] - cand_centers[sel]) / scene_scale
#                 dot = np.clip(np.dot(fwds[i], fwds[sel]), -1.0, 1.0)
#                 d_ang = np.arccos(dot) / np.pi
#                 d = pos_weight * d_pos + ang_weight * d_ang
#                 if d < min_dist: min_dist = d
            
#             # 综合打分
#             score = min_dist + sim_weight * norm_scores[i]
#             if score > best_score:
#                 best_score = score
#                 best_idx = i
        
#         if best_idx != -1:
#             selected_local.append(best_idx)
#         else:
#             break

#     # 输出
#     final_global_idx = [cand_indices[i] for i in selected_local]
#     final_names = [db_names[i] for i in final_global_idx]
    
#     # 写文件...
#     pairs_out = [(query_name, name) for name in final_names]
#     with open(output, "w") as f:
#         f.write("\n".join(" ".join([i, j]) for i, j in pairs_out))
        
#     return final_global_idx







