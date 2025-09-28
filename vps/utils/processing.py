import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, List, Dict
import os
import os.path as osp
from PIL import Image
import math
import torch
from torchvision import transforms

def load_imagesPathList_as_tensor(path_list, PIXEL_LIMIT=255000):
    """
    用于pi3的图像预处理
    直接从图片路径列表加载图片,resize到统一尺寸,转为[N, 3, H, W]的tensor
    Args:
        path_list: List[str]，图片路径列表
        PIXEL_LIMIT: int,单张图片最大像素数
    Returns:
        torch.Tensor: [N, 3, H, W]
    """
    sources = []
    for img_path in path_list:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

def generate_ref_list(
    query_img: Union[str, Path], 
    ref_dir: Union[str, Path], 
    pairs_file: Union[str, Path]) -> List[str]:
    """
    根据给定的查询图像，从配对文件中生成参考图像列表。
    """
    query_name = Path(query_img).name
    ref_list = []
    with open(pairs_file, 'r') as f:
        for line in f:
            A, ref_name = line.strip().split()
            A = Path(A).name
            if A != query_name:
                continue
            ref_name = Path(ref_name).name
            ref_image = Path(ref_dir) / "rgb" / ref_name
            if Path(ref_image).exists():
                ref_list.append(str(ref_image))
            ref_render = Path(ref_dir) / "rgb_render" / ref_name
            if Path(ref_render).exists():
                ref_list.append(str(ref_render))
    return ref_list

def compute_scale_factor( 
                           pred_depth: np.ndarray, 
                           gt_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> float:
    """
    计算模型预测深度和真值深度之间的尺度因子。
    
    Args:
        pred_depth: 预测的深度图 np.ndarray
        gt_depth: 真值深度图 np.ndarray
        mask: 可选的有效深度值掩码
        
    Returns:
        尺度因子
    """
    if pred_depth.shape != gt_depth.shape:
        gt_depth = cv2.resize(gt_depth, 
                                (pred_depth.shape[1], pred_depth.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    if mask is None:
        mask = (gt_depth > 1e-1) & (pred_depth > 1e-1)
    valid_pred = pred_depth[mask]
    valid_gt = gt_depth[mask]
    scale_factors = valid_gt / valid_pred
    if len(scale_factors) == 0:
        return 1.0
    else:
        return float(np.median(scale_factors))

def umeyama_alignment(src, dst):
    """
    src: Nx3 predicted points
    dst: Nx3 GT points
    Returns: s, R, t
    """
    assert src.shape == dst.shape
    n = src.shape[0]
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    
    cov = src_centered.T @ dst_centered / n
    
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    var_src = (src_centered ** 2).sum() / n
    s = np.sum(S) / var_src
    
    t = mu_dst - s * R @ mu_src
    
    return s, R, t

def motion_averaging(ref2query_poses, ref_poses, weights=None, outlier_threshold=0.3, min_direction_norm=0.001):
    """
    基于方向汇聚的运动平均算法,用于优化查询图像的pose
    
    思路:每个ref通过相对变换得到指向query的方向向量,
    多个方向汇聚到一个点(query的位置),用最小二乘法优化
    
    Args:
        ref2query_poses: List[np.ndarray] - 从参考图像到查询图像的相对变换矩阵 (4x4)
                        这些变换可能不准确且平移无尺度
        ref_poses: List[np.ndarray] - 参考图像的ground truth pose矩阵 (4x4)  
        weights: Optional[List[float]] - 每个方向的权重,如果为None则使用均等权重
        outlier_threshold: float - 异常值检测阈值，用于过滤垃圾朝向
        min_direction_norm: float - 最小方向向量模长，过小的认为是噪声
    
    Returns:
        np.ndarray: 优化后的查询图像pose矩阵 (4x4)
        List[bool]: 每个参考图像是否被使用（未被过滤掉）
    """
    if len(ref2query_poses) != len(ref_poses):
        raise ValueError("相对变换数量必须与参考pose数量相等")
    
    if len(ref2query_poses) < 2:
        raise ValueError("至少需要2个参考图像进行运动平均")
    
    n_pairs = len(ref2query_poses)
    
    # 步骤1: 提取方向向量和参考位置
    directions = []  # 归一化的方向向量
    ref_positions = []  # 参考图像的位置
    valid_indices = []  # 有效的索引
    
    for i, (ref2query_pose, ref_pose) in enumerate(zip(ref2query_poses, ref_poses)):
        # 提取平移向量（ref指向query的方向）
        translation = ref2query_pose[:3, 3]
        translation_norm = np.linalg.norm(translation)
        
        # 过滤太小的平移（可能是噪声）
        if translation_norm < min_direction_norm:
            print(f"过滤参考图像{i}: 平移向量太小 ({translation_norm:.6f})")
            continue
            
        # 归一化得到方向向量
        direction = translation / translation_norm
        directions.append(direction)
        
        # 参考图像的位置
        ref_position = ref_pose[:3, 3]
        ref_positions.append(ref_position)
        
        valid_indices.append(i)
    
    if len(directions) < 2:
        raise ValueError("有效的参考图像不足2个,无法进行运动平均")
    
    directions = np.array(directions)
    ref_positions = np.array(ref_positions)
    
    print(f"使用{len(directions)}个有效方向进行运动平均")
    
    # 步骤2: 旋转平均（使用有效的相对变换）
    valid_rotations = []
    valid_ref_poses = []
    for i in valid_indices:
        valid_rotations.append(ref2query_poses[i][:3, :3])
        valid_ref_poses.append(ref_poses[i])
    
    query_rotation = _average_rotations_geometric(valid_rotations, valid_ref_poses)
    
    # 步骤3: 过滤异常方向
    if outlier_threshold > 0:
        directions, ref_positions, inlier_mask = _filter_outlier_directions(
            directions, ref_positions, outlier_threshold
        )
        print(f"异常值过滤后剩余{len(directions)}个方向")
    else:
        inlier_mask = np.ones(len(directions), dtype=bool)
    
    # 步骤4: 最小二乘法求解query位置
    query_position = _solve_position_least_squares(directions, ref_positions, weights)
    
    # 构建最终的4x4变换矩阵
    query_pose = np.eye(4)
    query_pose[:3, :3] = query_rotation
    query_pose[:3, 3] = query_position
    
    # 构建使用标记
    used_mask = np.zeros(n_pairs, dtype=bool)
    valid_inlier_indices = np.array(valid_indices)[inlier_mask]
    used_mask[valid_inlier_indices] = True
    
    return query_pose, used_mask.tolist()


def _average_rotations_geometric(ref2query_rotations, ref_poses):
    """
    几何方法的旋转平均
    从ref到query的旋转,结合ref的旋转,计算query的绝对旋转
    """
    from scipy.spatial.transform import Rotation as R
    
    # 计算每个候选的查询旋转
    candidate_rotations = []
    
    for R_rel, ref_pose in zip(ref2query_rotations, ref_poses):
        R_ref = ref_pose[:3, :3]
        # query的绝对旋转 = ref的绝对旋转 @ ref到query的相对旋转
        R_query_candidate = R_ref @ R_rel
        candidate_rotations.append(R_query_candidate)
    
    # 转换为四元数进行平均
    quaternions = []
    for R_mat in candidate_rotations:
        quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        quaternions.append(quat)
    
    quaternions = np.array(quaternions)
    
    # 确保所有四元数在同一半球（避免符号模糊）
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[i]) < 0:
            quaternions[i] *= -1
    
    # 简单平均（也可以用加权平均）
    avg_quat = np.mean(quaternions, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)  # 归一化
    
    # 转换回旋转矩阵
    return R.from_quat(avg_quat).as_matrix()


def _filter_outlier_directions(directions, ref_positions, threshold):
    """
    过滤异常的方向向量
    
    Args:
        directions: np.ndarray (N, 3) - 归一化的方向向量
        ref_positions: np.ndarray (N, 3) - 参考位置
        threshold: float - 异常值阈值
    
    Returns:
        filtered_directions: np.ndarray - 过滤后的方向
        filtered_ref_positions: np.ndarray - 过滤后的参考位置  
        inlier_mask: np.ndarray - 内点掩码
    """
    n_directions = len(directions)
    if n_directions <= 2:
        # 方向太少，不进行过滤
        return directions, ref_positions, np.ones(n_directions, dtype=bool)
    
    # 计算每个方向与其他方向的平均夹角
    angle_scores = []
    
    for i in range(n_directions):
        angles = []
        for j in range(n_directions):
            if i != j:
                # 计算方向向量之间的夹角
                cos_angle = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # 使用中位数角度作为该方向的一致性评分
        median_angle = np.median(angles)
        angle_scores.append(median_angle)
    
    angle_scores = np.array(angle_scores)
    
    # 过滤掉角度偏差过大的方向
    # 使用robust统计方法
    q75 = np.percentile(angle_scores, 75)
    inlier_mask = angle_scores <= (q75 + threshold)
    
    print(f"方向一致性评分: {angle_scores}")
    print(f"内点掩码: {inlier_mask}")
    
    return directions[inlier_mask], ref_positions[inlier_mask], inlier_mask


def _solve_position_least_squares(directions, ref_positions, weights=None):
    """
    用最小二乘法求解查询位置
    
    每个约束:query_position = ref_position + scale * direction
    我们要同时求解query_position和各个scale
    
    Args:
        directions: np.ndarray (N, 3) - 归一化的方向向量
        ref_positions: np.ndarray (N, 3) - 参考位置
        weights: Optional[np.ndarray] - 权重
        
    Returns:
        np.ndarray: 查询位置 (3,)
    """
    n = len(directions)
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
        if len(weights) != n:
            weights = np.ones(n)
    
    # 构建线性方程组
    # 变量: [query_x, query_y, query_z, scale_1, scale_2, ..., scale_n]
    # 约束: query_position - scale_i * direction_i = ref_position_i
    
    # 系数矩阵 A 和右端向量 b
    # 每个参考点贡献3个方程（x, y, z）
    A = np.zeros((3 * n, 3 + n))
    b = np.zeros(3 * n)
    
    for i in range(n):
        row_start = 3 * i
        
        # query_position 的系数
        A[row_start:row_start+3, :3] = np.eye(3) * weights[i]
        
        # scale_i 的系数
        A[row_start:row_start+3, 3+i] = -directions[i] * weights[i]
        
        # 右端向量
        b[row_start:row_start+3] = ref_positions[i] * weights[i]
    
    # 求解最小二乘问题
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        query_position = solution[:3]
        scales = solution[3:]
        
        print(f"估计的尺度: {scales}")
        return query_position
        
    except np.linalg.LinAlgError:
        # 如果最小二乘失败，使用简单的中心化方法
        print("最小二乘求解失败，使用简单平均方法")
        
        # 每个ref_position + 平均距离 * direction 的加权平均
        estimated_positions = []
        avg_distance = 5.0  # 假设平均距离
        
        for i in range(n):
            estimated_pos = ref_positions[i] + avg_distance * directions[i]
            estimated_positions.append(estimated_pos)
        
        estimated_positions = np.array(estimated_positions)
        query_position = np.average(estimated_positions, axis=0, weights=weights)
        
        return query_position


def example_motion_averaging_usage():
    """
    Motion Averaging使用示例 - 基于方向汇聚的方法
    """
    # 示例数据：假设有3个参考图像
    # 参考图像的ground truth poses (4x4矩阵)
    ref_poses = [
        np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0], 
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [0.707, -0.707, 0.0, 4.0],
            [0.707, 0.707, 0.0, 5.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [0.0, 1.0, 0.0, 7.0],
            [-1.0, 0.0, 0.0, 8.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    ]
    
    # 模型推理得到的从ref到query的相对变换（可能有噪声且平移无尺度）
    ref2query_poses = [
        np.array([
            [0.9, 0.1, 0.0, 2.5],   # ref指向query的方向（无尺度）
            [-0.1, 0.9, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [0.8, 0.6, 0.0, -1.2],  # 另一个方向
            [-0.6, 0.8, 0.0, -0.8],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [0.7, 0.7, 0.0, -3.1],  # 第三个方向
            [-0.7, 0.71, 0.0, -2.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    ]
    
    # 可选：为每个相对变换分配权重（基于置信度）
    weights = [0.8, 0.9, 0.7]  # 根据匹配质量或其他指标确定
    
    # 执行基于方向汇聚的运动平均
    try:
        optimized_query_pose, used_mask = motion_averaging(
            ref2query_poses=ref2query_poses,
            ref_poses=ref_poses,
            weights=weights,
            outlier_threshold=0.3,  # 异常值过滤阈值
            min_direction_norm=0.1  # 最小方向向量模长
        )
        
        print("优化后的查询图像pose:")
        print(optimized_query_pose)
        print()
        print("使用的参考图像:")
        for i, used in enumerate(used_mask):
            status = "✓" if used else "✗"
            print(f"  参考图像 {i+1}: {status}")
        
        return optimized_query_pose, used_mask
        
    except Exception as e:
        print(f"Motion averaging执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_motion_averaging(ref_poses, ref2query_poses, query_pose_estimated, used_mask):
    """
    可视化motion averaging的结果
    
    Args:
        ref_poses: List[np.ndarray] - 参考pose列表
        ref2query_poses: List[np.ndarray] - ref到query的相对变换列表
        query_pose_estimated: np.ndarray - 估计的query pose
        used_mask: List[bool] - 使用的参考图像掩码
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制参考图像位置
        ref_positions = np.array([pose[:3, 3] for pose in ref_poses])
        ax.scatter(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], 
                  c='blue', s=100, label='参考图像', marker='s')
        
        # 绘制方向向量
        for i, (ref_pose, ref2query_pose, used) in enumerate(zip(ref_poses, ref2query_poses, used_mask)):
            if not used:
                continue
                
            ref_pos = ref_pose[:3, 3]
            direction = ref2query_pose[:3, 3]
            direction_norm = direction / np.linalg.norm(direction)
            
            # 绘制方向向量（箭头）
            ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2],
                     direction_norm[0], direction_norm[1], direction_norm[2],
                     length=2.0, color='red', alpha=0.7, arrow_length_ratio=0.1)
        
        # 绘制估计的query位置
        query_pos = query_pose_estimated[:3, 3]
        ax.scatter(query_pos[0], query_pos[1], query_pos[2], 
                  c='green', s=200, label='估计query位置', marker='*')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Motion Averaging 方向汇聚可视化')
        
        plt.show()
        
    except ImportError:
        print("matplotlib未安装,无法进行可视化")
        return
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        return


def compute_motion_averaging_error(query_pose_estimated, query_pose_gt):
    """
    计算motion averaging的误差
    
    Args:
        query_pose_estimated: np.ndarray - 估计的query pose (4x4)
        query_pose_gt: np.ndarray - ground truth query pose (4x4)
        
    Returns:
        dict: 包含各种误差指标
    """
    # 位置误差
    position_error = np.linalg.norm(
        query_pose_estimated[:3, 3] - query_pose_gt[:3, 3]
    )
    
    # 旋转误差（使用角度差）
    from scipy.spatial.transform import Rotation as R
    
    R_est = R.from_matrix(query_pose_estimated[:3, :3])
    R_gt = R.from_matrix(query_pose_gt[:3, :3])
    
    # 计算相对旋转
    R_diff = R_gt.inv() * R_est
    rotation_error_rad = R_diff.magnitude()
    rotation_error_deg = np.degrees(rotation_error_rad)
    
    return {
        'position_error': position_error,
        'rotation_error_rad': rotation_error_rad,
        'rotation_error_deg': rotation_error_deg,
        'pose_matrix_error': np.linalg.norm(query_pose_estimated - query_pose_gt)
    }
