from flask import config
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import cv2
import json
import os
import time
import logging
import torch.nn.functional as F
import open3d as o3d
from depth_anything_3.api import DepthAnything3
from vps.utils.processing import compute_scale_factor, generate_ref_list, unproject_depth_map_to_point_cloud, trans_point_cloud
from vps.utils.find_similar import get_descriptors, parse_names
from vps.utils.motion_averaging_raw import MotionAveraging
from vps.nav.point2map import *
from torch.nn.attention import sdpa_kernel, SDPBackend

#DA3模型直接输出camera是w2c
class PoseEstimator:  
    """Pose estimation module using DA3."""
    
    def __init__(self, config: Dict):
        """
        Initialize the pose estimator.
        
        Args:
            config: Configuration dictionary containing pose estimation settings
        """
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # Set up dtype
        if config['system']['dtype'] == 'float16':
            self.dtype = torch.float16
        elif config['system']['dtype'] == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
            
        self.model = DepthAnything3.from_pretrained(config['pose']['da3']['model_path']).to("cuda")
        self.model = self.model.to(self.device)
    

        #nav settings
        self.depth_nav = config['depth_nav']['able']
        self.up =  config['depth_nav']['height_up']
        self.cam_real_h = float(config['depth_nav']['camera_real_h'])
        self.map_path = Path(config['depth_nav']['map_path'])
        self.yaml_path = Path(config['depth_nav']['yaml_path'])
        self.min_dist = float(config['depth_nav']['min_dist'])
        self.max_dist = float(config['depth_nav']['max_dist'])

    def save_points_as_ply(self, points: np.ndarray, filename: str):
        # 转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # 保存为 PLY 文件
        o3d.io.write_point_cloud(filename, pcd)
        print(f"✅ Saved {len(points)} points to {filename}")

    def generate_mask_from_coord(
        self, # 假设这是您类的方法
        original_coord: torch.Tensor, 
        load_size: int, 
        target_size: int
    ) -> np.ndarray: # 修改为 np.ndarray
        """
        根据原始图像坐标信息，生成一个在 target_size 空间中的布尔 Mask。
        
        Args:
            original_coord (torch.Tensor): 形状 (6) 的张量，包含 [x1, y1, x2, y2, W_orig, H_orig]。
            load_size (int): 填充后、未最终缩放的方形图像尺寸 (self.image_load_size)。
            target_size (int): 最终的模型输入尺寸 (例如 518)。

        Returns:
            np.ndarray: 形状 (target_size, target_size) 的布尔 Mask (NumPy 数组)。
        """
        # 只需要一个单独的 Mask，不需要列表 final_mask

        # 1. 提取原始尺寸 W_orig, H_orig
        coord = original_coord.cpu().numpy()
        # original_coord 是 [x1, y1, x2, y2, W_orig, H_orig]
        W_orig, H_orig = coord[4], coord[5]
        
        # 2. 计算填充值 (与 load_and_preprocess_images_square 保持一致)
        max_dim = max(W_orig, H_orig)
        left_pad = (max_dim - W_orig) // 2
        top_pad = (max_dim - H_orig) // 2
        
        # 3. 计算在 load_size 空间中的像素边界
        scale_factor = load_size / max_dim
        
        # 使用 floor/ceil 确保包含所有边界像素
        x_start = int(np.floor(left_pad * scale_factor))
        y_start = int(np.floor(top_pad * scale_factor))
        
        x_end = int(np.ceil((left_pad + W_orig) * scale_factor))
        y_end = int(np.ceil((top_pad + H_orig) * scale_factor))
        
        # 4. 创建 load_size 尺寸的初始 Mask
        mask_load_size = torch.zeros((load_size, load_size), dtype=torch.bool)
        
        # 标记原始内容区域 (确保索引在界限内)
        x_end = min(x_end, load_size)
        y_end = min(y_end, load_size)
        
        if x_end > x_start and y_end > y_start:
            mask_load_size[y_start:y_end, x_start:x_end] = True

        # 5. 缩放 Mask 到目标尺寸 (target_size x target_size)
        
        # [H, W] -> [1, 1, H, W] (用于 F.interpolate)
        mask_load_size = mask_load_size.float().unsqueeze(0).unsqueeze(0)
        
        mask_target_size = F.interpolate(
            mask_load_size, 
            size=(target_size, target_size), 
            mode="bilinear", 
            align_corners=False
        ).squeeze() # 最终形状是 [target_size, target_size]
        
        # 6. 二值化并返回 NumPy 数组
        # 使用 > 0.5 进行二值化，转换为布尔类型
        final_mask_bool_tensor = mask_target_size > 0.5
        
        # 转换成 NumPy 数组
        return final_mask_bool_tensor.cpu().numpy()

    
    def estimate_pose(
        self, 
        query_img: Union[str, Path],
        query_depth: Optional[Union[str, Path]] = None
        ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Estimate relative pose between query and reference images.
        
        Args:
            query_img: Path to query image
            query_depth: Optional ground truth depth for scale recovery
        Returns:
            Tuple containing:
            - Absoulte pose matrix (4x4)  camera2w
            - Depth map
            - Occupancy map
        """
        # Preprocess images
        # [query,ref1,ref2,ref3.....]
        # 加载和预处理图像
        image_paths = []
        query_img = Path(query_img)
        image_paths.append(query_img)
        ref_imgs = generate_ref_list(query_img, self.config['vpr']['ref_data_path'], self.config['vpr']['pairs_file_path'])
        image_paths.extend(ref_imgs)
        image_paths = [str(p) for p in image_paths]
        logging.info(f"image数量: {len(image_paths)}")
        assert len(image_paths) >=2
        start_time = time.time()
        prediction = self.model.inference(
                image=image_paths,
                export_dir="",
                export_format="mini_npz"
            )
        end_time = time.time()
        extrinsic = prediction.extrinsics  #3x4 w2c
        depth_map = prediction.depth
        intrinsic = prediction.intrinsics
        logging.info(f"DA3 inference time: {end_time - start_time:.2f}s")
        # Compute relative pose
        P_query = np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0) #w2c
        # 读取参考图像s的pose
        ref_poses_gt = [np.loadtxt(Path(ref_img).parent.parent / "poses" / f"{Path(ref_img).stem}.txt").reshape(4, 4) for ref_img in ref_imgs] #c2w
        final_pose = None
        final_depth = None
        P_query = np.linalg.inv(np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0)) #c2w
        P_refs = []
        extrinsic_ref = extrinsic[1:] 
        for ext in extrinsic_ref:
            P_refs.append(np.linalg.inv(np.concatenate([ext, np.array([[0, 0, 0, 1]])], axis=0))) #c2w
        ma = MotionAveraging()
        q2r_poses = [np.linalg.inv(P_ref) @ P_query for P_ref in P_refs]
        # r2q_poses = [np.linalg.inv(P_query) @ P_ref for P_ref in P_refs]
        for q2r_pose in q2r_poses:
            q2r_pose[0:3,3] = q2r_pose[0:3,3] / np.linalg.norm(q2r_pose[0:3,3])
        # 计算最终的位姿
        final_pose = ma.motion_averaging(ref_poses_gt, q2r_poses)
        logging.info(f"start analyze scale:")
        scales = []
        for i in range(len(ref_poses_gt)):
            # --- 真实世界的相对位移 (GT) ---
            # P_query_gt 是你通过 Motion Averaging 得到的 final_pose (c2w)
            # P_ref_gt 是 ref_poses_gt[i] (c2w)
            # 计算从 Ref 到 Query 的位移向量
            t_ref2query_gt = final_pose[:3, 3] - ref_poses_gt[i][:3, 3]
            dist_gt = np.linalg.norm(t_ref2query_gt)

            # --- 模型预测的相对位移 (Model Space) ---
            # 这里的 P_query 和 P_refs 是模型输出的 c2w
            t_ref2query_model = P_query[:3, 3] - P_refs[i][:3, 3]
            dist_model = np.linalg.norm(t_ref2query_model)

            if dist_model > 1e-6: # 防止除以0
                scales.append(dist_gt / dist_model)

        # 2. 取中位数得到最终尺度因子 s
        final_scale = np.mean(scales)

        logging.info(f"检测到的尺度因子 s = {scales}")




        final_depth = depth_map[0].squeeze()
        result_path = Path(self.config['pose']['da3']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)
        np.savetxt(result_path.parent.parent/ f"last_pose.txt", final_pose)
        logging.info(f"da3_final_pose: {final_pose}")
        new_map = None
        if self.depth_nav == True:
            point_start_time = time.time()

            # pcd = unproject_depth_map_to_point_cloud(depth_map=final_depth,intrinsic_cam=intrinsic[0],extrinsic_cam=final_pose)
            all_points = []
            for i in range(len(extrinsic)):
                pcd = unproject_depth_map_to_point_cloud(depth_map=depth_map[i].squeeze(),intrinsic_cam=intrinsic[i],
                               extrinsic_cam=np.linalg.inv(np.concatenate([extrinsic[i], np.array([[0, 0, 0, 1]])], axis=0)))
                all_points.append(pcd.reshape(-1, 3))
            all_points = np.concatenate(all_points, axis=0)
            #   c2w  @  w2c =
            pcd = trans_point_cloud(all_points,extrinsic_cam=final_pose @ np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0))



            if self.up == 'z': 
                cam_pred_h = final_pose[2][3]#z-up
            else:
                cam_pred_h = final_pose[1][3]#y-up
            logging.info(cam_pred_h)
            pred_floor_h = get_floor_height(pcd=pcd,cam_pred_h=cam_pred_h,up=self.up)
            logging.info(pred_floor_h)
            scale = get_height_scale(cam_pred_h,pred_floor_h,self.cam_real_h)
            logging.info(f"camera_floor scale is {scale}")
            pcd = unproject_depth_map_to_point_cloud(depth_map=final_depth,intrinsic_cam=intrinsic[0],extrinsic_cam=final_pose,scale=scale)
            # pcd = trans_point_cloud(final_point,extrinsic_cam=final_pose,scale=scale)

            # pred_floor_h = get_floor_height(pcd,cam_pred_h,up=self.up)
            pred_floor_h = cam_pred_h - scale * (cam_pred_h - pred_floor_h)
            obstacle_points, avalibale_points = segment_points_h(pcd,pred_floor_h,cam_pred_h,up=self.up)

            if self.map_path.exists() and self.yaml_path.exists():
                new_map = get_new_occupancy_map(
                    obs_points=obstacle_points,
                    ava_points=avalibale_points,
                    map_path=self.map_path,
                    yaml_path=self.yaml_path,
                    camera_6dpose=final_pose,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist,
                    occupancy_min_points_per_cell=15,
                    up=self.up,
                    showself=True
                )

                logging.info("Depth navigation path executed and map updated.")
            else:
                logging.warning(
                    f"⚠️ 跳过地图更新：地图文件或 YAML 文件不存在。"
                    f"Map Path Exists: {self.map_path.exists()}, "
                    f"YAML Path Exists: {self.yaml_path.exists()}"
                )
            point_end_time = time.time()
            logging.info(f"point time: {point_end_time - point_start_time:.4f}s")
            self.save_points_as_ply(pcd, "output.ply")

            cv2.imwrite('dawdawd.png',new_map)
        
        # np.save('a.npy',final_depth)
        # self.visualize_depth_map_colored_cv2(final_depth, 'colored_final_depth_cv2.png')
        # self.visualize_depth_map_colored_cv2(depth_map[0].squeeze(),'colored_final_depth_cv3.png')



        
        return (
        final_pose,                   # 1. Absoulte pose matrix (4x4)
        final_depth,                  # 2. Depth map (假设取第一个查询图，并降维)
        new_map                       # 3. New Occupancy Map (Optional, np.ndarray or None)
    ) 
    def visualize_depth_map_colored_cv2(self,depth_data: np.ndarray, output_path: str):
        """
        使用 OpenCV 加载深度图，应用颜色映射并保存为彩色图片。

        Args:
            depth_data (np.ndarray): 输入的深度图 NumPy 数组 (例如 final_depth)
            output_path (str): 彩色图像的保存路径 (e.g., 'colored_depth_cv2.png')
        """
        # 1. 处理 NaN 值 (如果您的深度图中有 NaN)
        # 将 NaN 替换为 0。或者，您可以选择一个非常小的负数或最大值+1来区分
        # 如果您希望 NaN 区域显示为黑色，替换为 0 是合适的。
        depth_data_processed = np.nan_to_num(depth_data, nan=0.0) 
        
        # 2. 归一化深度值到 0-255 范围
        # 找到有效（非零）深度值的范围进行归一化，避免 0 值（填充）影响整体颜色映射
        non_zero_depths = depth_data_processed[depth_data_processed > 0]
        
        if non_zero_depths.size == 0:
            print("警告：深度图中没有有效的（非零）数据。保存全黑图像。")
            colored_depth_image = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)
        else:
            min_val = np.min(non_zero_depths)
            max_val = np.max(non_zero_depths)
            
            if max_val == min_val:
                # 所有有效深度值都相同，统一设为中间值
                normalized_depth = np.full_like(depth_data_processed, 127, dtype=np.uint8)
            else:
                # 归一化到 0-255
                # 注意：只对 > 0 的部分进行缩放，0 保持为 0 (黑色)
                normalized_depth = np.zeros_like(depth_data_processed, dtype=np.uint8)
                scale_factor = 255.0 / (max_val - min_val)
                normalized_depth[depth_data_processed > 0] = (
                    (depth_data_processed[depth_data_processed > 0] - min_val) * scale_factor
                ).astype(np.uint8)
            
            # 3. 应用颜色映射
            # OpenCV 提供了多种颜色映射：COLORMAP_JET, COLORMAP_MAGMA, COLORMAP_VIRIDIS 等
            colored_depth_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        # 4. 保存彩色图像
        cv2.imwrite(output_path, colored_depth_image)
        
        print(f"✅ 彩色深度图已成功保存到: {output_path}")