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
from scipy.spatial.transform import Rotation as R
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..utils.processing import compute_scale_factor, generate_ref_list
from ..utils.logging_error import logging_error
from vps.utils.motion_averaging_raw import MotionAveraging
#VGGT模型输出camera是c2w
class PoseEstimatorVGGT:  
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
            self.dtype = torch.bfloat16
            
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
        # [query,ref1,ref2,ref3.....]
        # 加载和预处理图像
        image_paths = []
        query_img = Path(query_img)
        image_paths.append(query_img)
        ref_imgs = generate_ref_list(query_img, self.config['vpr']['ref_data_path'], self.config['vpr']['pairs_file_path'])
        image_paths.extend(ref_imgs)
        logging.info(f"image数量: {len(image_paths)}")
        assert len(image_paths) >=2
        start_time = time.time()
        images, original_coords = load_and_preprocess_images_square(image_paths, self.image_size)
        images = images.to(self.device)
        original_coords = original_coords.to(original_coords.device)
        
        # 运行VGGT获取相机参数和深度图
        extrinsic, intrinsic, depth_map, depth_conf = self.run_VGGT(self.model, images, self.dtype, 518)
        # model_pred_pose = [np.linalg.inv(np.concatenate([a, np.array([[0, 0, 0, 1]])], axis=0)) for a in extrinsic]
        end_time = time.time()
        logging.info(f"VGGT 运行时间: {end_time - start_time:.2f}s")


        ref_poses_gt = [np.loadtxt(Path(ref_img).parent.parent / "poses" / f"{Path(ref_img).stem}.txt").reshape(4, 4) for ref_img in ref_imgs] #c2w
############################################################最相似的ref充当锚点########################################
        P_query = np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0) #w2c
        P_ref = np.concatenate([extrinsic[1], np.array([[0, 0, 0, 1]])], axis=0) #w2c
        query2ref = P_ref @ np.linalg.inv(P_query)
        # 读取参考图像的pose
        # 最相似的ref充当锚点
        ref_img = Path(ref_imgs[0])
        final_pose = ref_poses_gt[0] @ query2ref
        result_path = Path(self.config['pose']['vggt']['results_dir']+"最相似") / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)

############################################################简单求平均################################################
        P_query = np.linalg.inv(np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0)) #c2w
        P_refs = []
        exts = extrinsic[1:]
        for ext in exts:
            P_refs.append(np.linalg.inv(np.concatenate([ext, np.array([[0, 0, 0, 1]])], axis=0))) #c2w
        query2ref = [np.linalg.inv(P_ref) @ P_query for P_ref in P_refs]

        # 1. 收集所有参考帧推算出的候选位姿 (Candidate Poses)
        candidate_poses = []
        for i in range(len(P_refs)):
            # T_q2w = T_ref2w * T_q2ref
            t_q2w = ref_poses_gt[i] @ query2ref[i]
            candidate_poses.append(t_q2w)

        # 2. 对平移部分 (Translation) 求简单算术平均
        all_translations = np.array([p[:3, 3] for p in candidate_poses])
        avg_translation = np.mean(all_translations, axis=0)

        # 3. 对旋转部分 (Rotation) 求平均
        # 注意：旋转不能直接相加除以N，需要用四元数平均或SVD方法
        all_rotations = [p[:3, :3] for p in candidate_poses]
        rots = R.from_matrix(all_rotations)
        # scipy的mean方法实现了基于四元数的旋转平均
        avg_rotation_matrix = rots.mean().as_matrix()

        # 4. 组合成最终的 4x4 矩阵
        final_pose = np.eye(4)
        final_pose[:3, :3] = avg_rotation_matrix
        final_pose[:3, 3] = avg_translation
        result_path = Path(self.config['pose']['vggt']['results_dir']+"简单平均") / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)


############################################################运动平均##################################################
        scale_factor = 1.0
        if self.config['pose']['use_motion_average']:
            logging.info("使用运动平均")
            P_query = np.linalg.inv(np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0)) #c2w
            P_refs = []
            exts = extrinsic[1:] 
            for ext in exts:
                P_refs.append(np.linalg.inv(np.concatenate([ext, np.array([[0, 0, 0, 1]])], axis=0))) #c2w
            ma = MotionAveraging()
            q2r_poses = [np.linalg.inv(P_ref) @ P_query for P_ref in P_refs]
            # r2q_poses = [np.linalg.inv(P_query) @ P_ref for P_ref in P_refs]
            for q2r_pose in q2r_poses:
                q2r_pose[0:3,3] = q2r_pose[0:3,3] / np.linalg.norm(q2r_pose[0:3,3])
            final_pose = ma.motion_averaging(ref_poses_gt, q2r_poses)

        # else:
        #     if query_depth is not None:
        #         if Path(query_depth).exists():
        #             logging.info(f"query_depth 存在")
        #             vggt_depth = depth_map[-1].squeeze()
        #             query_depth = np.load(query_depth)
        #             query_depth = cv2.resize(query_depth, (vggt_depth.shape[1], vggt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        #             scale_factor = compute_scale_factor(vggt_depth, query_depth)
        #     else:
        #         ref_depth = None
        #         if Path(ref_img.parent.parent/"depth"/f"{ref_img.stem}.npy").exists():
        #             ref_depth = np.load(ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy")
        #         if Path(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy").exists():
        #             ref_depth = np.load(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy")
        #         if ref_depth is not None:
        #             vggt_depth = depth_map[0].squeeze()
        #             ref_depth = cv2.resize(ref_depth, (vggt_depth.shape[1], vggt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        #             scale_factor = compute_scale_factor(vggt_depth, ref_depth)
        #     query2ref[:3, 3] *= scale_factor
        #     logging.info(f"depth pred scale_factor: {scale_factor}")
        #     final_pose = ref_poses_gt[0] @ query2ref
        #  计算最终的位姿
        
        result_path = Path(self.config['pose']['vggt']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)
        np.savetxt(result_path.parent.parent/ f"last_pose.txt", final_pose)
        logging.info(f"vggt_final_pose: {final_pose}")




        # q_gt_pose = np.loadtxt(Path(ref_img).parent.parent.parent/"test" / "poses" / f"{Path(query_img).stem}.txt").reshape(4, 4)#query2world
        # logging_error(model_pred_pose=model_pred_pose,final_pose=final_pose,q_gt_pose=q_gt_pose,refs_gt_pose=ref_poses_gt)
        depth = None
        map = None
        return final_pose,depth,map