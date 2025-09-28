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
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..utils.processing import compute_scale_factor, generate_ref_list
from ..utils.find_similar import get_descriptors, parse_names
class PoseEstimator:
    """Pose estimation module using VGGT."""
    
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
            self.dtype = torch.float32
            
        # Initialize VGGT model
        self.model = VGGT()
        model_path = config['pose']['vggt']['model_path']
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Image preprocessing settings
        self.image_size = config['pose']['vggt']['image_size']

    def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """将深度图反投影为点云，形状 [N, 3]"""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth.reshape(-1)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        pts = np.stack([x, y, z], axis=1)
        valid = z > 0
        return pts[valid]

    def run_VGGT(self, model, images, dtype, resolution=518):
    # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # hard-coded to use 518 for VGGT
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        return extrinsic, intrinsic, depth_map, depth_conf
    
    def estimate_pose(
        self, 
        query_img: Union[str, Path],
        query_depth: Optional[Union[str, Path]] = None
        ) -> np.ndarray:
        """
        Estimate relative pose between query and reference images.
        
        Args:
            query_img: Path to query image
            query_depth: Optional ground truth depth for scale recovery
        Returns:
            Tuple containing:
            - Absoulte pose matrix (4x4)  camera2w
            - Depth map
            - Depth confidence map
            - Scale factor (if gt_depth provided, else 1.0)
        """
        # Preprocess images
        # [ref1,ref2,ref3.....,query]
        # 加载和预处理图像
        image_paths = []
        query_img = Path(query_img)
        image_paths.append(query_img)
        ref_imgs = generate_ref_list(query_img, self.config['pose']['vggt']['ref_dir'], self.config['vpr']['pairs_file_path'])
        image_paths.extend(ref_imgs)
        logging.info(f"image数量: {len(image_paths)}")
        assert len(image_paths) >=2
        start_time = time.time()
        images, original_coords = load_and_preprocess_images_square(image_paths, self.image_size)
        images = images.to(self.device)
        original_coords = original_coords.to(original_coords.device)
        
        # 运行VGGT获取相机参数和深度图
        extrinsic, intrinsic, depth_map, depth_conf = self.run_VGGT(self.model, images, self.dtype, 518)
        # print(f"intrinsic: {intrinsic}")
        end_time = time.time()
        logging.info(f"VGGT 运行时间: {end_time - start_time:.2f}s")
        # Compute relative pose
        P_query = np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0) #w2c
        P_ref = np.concatenate([extrinsic[1], np.array([[0, 0, 0, 1]])], axis=0) #w2c
        query2ref = P_ref @ np.linalg.inv(P_query)

        # 读取第一张ref图像的pose
        ref_img = Path(ref_imgs[0])
        ref_pose = np.loadtxt(ref_img.parent.parent / "poses" / f"{ref_img.stem}.txt").reshape(4, 4) #c2w
        


        scale_factor = 1.0
        if query_depth is not None:
            if Path(query_depth).exists():
                logging.info(f"query_depth 存在")
                vggt_depth = depth_map[-1].squeeze()
                query_depth = np.load(query_depth)
                query_depth = cv2.resize(query_depth, (vggt_depth.shape[1], vggt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
                scale_factor = compute_scale_factor(vggt_depth, query_depth)
        else:
            ref_depth = None
            if Path(ref_img.parent.parent/"depth"/f"{ref_img.stem}.npy").exists():
                ref_depth = np.load(ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy")
            if Path(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy").exists():
                ref_depth = np.load(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy")
            if ref_depth is not None:
                vggt_depth = depth_map[0].squeeze()
                ref_depth = cv2.resize(ref_depth, (vggt_depth.shape[1], vggt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
                scale_factor = compute_scale_factor(vggt_depth, ref_depth)
        query2ref[:3, 3] *= scale_factor
        logging.info(f"scale_factor: {scale_factor}")
        
         # 计算最终的位姿
        final_pose = ref_pose @ query2ref
        result_path = Path(self.config['pose']['vggt']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)
        np.savetxt(result_path.parent.parent/ f"last_pose.txt", final_pose)
        logging.info(f"vggt_final_pose: {final_pose}")
        return final_pose 