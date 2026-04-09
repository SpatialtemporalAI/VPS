import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import cv2
import json
import logging
import sys
import time
sys.path.append("third_party/Pi3")
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function
from pi3.utils.geometry import depth_edge
from scipy.spatial.transform import Rotation as R
from vps.utils.processing import compute_scale_factor, generate_ref_list, load_imagesPathList_as_tensor
from vps.utils.motion_averaging_raw import MotionAveraging
from safetensors.torch import load_file
            

##pi3模型直接推理出的的pose是c2w
class PoseEstimatorPi3:
    """Pose estimation module using Pi3."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Pi3 pose estimator.
        
        Args:
            config: Configuration dictionary containing pose estimation settings
        """
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # Initialize Pi3 model

        self.model = Pi3().to(self.device).eval()
        weight_path = config['pose']['pi3']['model_path']
        weight = load_file(weight_path)
        self.model.load_state_dict(weight)



    def run_Pi3(self, model, images_path):
        imgs = load_imagesPathList_as_tensor(images_path).to(self.device)
        # --- Inference ---
        # Use mixed precision for better performance on compatible GPUs
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(imgs[None]) # Add batch dimension
        # Access outputs: results['points'], results['camera_poses'] and results['local_points'].
        return res
    
    def estimate_pose(
        self, 
        query_img: Union[str, Path],
        query_depth: Optional[Union[str, Path]] = None
        ):
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
        ref_imgs = generate_ref_list(query_img, self.config['pose']['pi3']['ref_dir'], self.config['vpr']['pairs_file_path'])
        image_paths.extend(ref_imgs)
        logging.info(f"image数量: {len(image_paths)}")
        assert len(image_paths) >=2
        start_time = time.time()
        results = self.run_Pi3(self.model, image_paths)#results['points'], results['camera_poses'] and results['local_points'].
        end_time = time.time()
        logging.info(f"Pi3 运行时间: {end_time - start_time:.2f}s")
        masks = torch.sigmoid(results['conf'][..., 0]) > 0.2
        non_edge = ~depth_edge(results['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        query_mask = masks[0].cpu().numpy()
        ref_mask = masks[1].cpu().numpy()


        Pred = results['camera_poses'][0].cpu().numpy() #c2w
        P_query = Pred[0] #c2w
        P_refs = Pred[1:] #c2w

        ref_poses_gt = [np.loadtxt(Path(ref_img).parent.parent / "poses" / f"{Path(ref_img).stem}.txt").reshape(4, 4) for ref_img in ref_imgs] #c2w

        scale_factor = 1.0
        final_pose = None
        if self.config['pose']['use_motion_average']:
        ############################################################最相似的ref充当锚点########################################
            query2ref = np.linalg.inv(P_refs[0]) @ P_query
            # 读取参考图像的pose
            # 最相似的ref充当锚点
            ref_img = Path(ref_imgs[0])
            final_pose = ref_poses_gt[0] @ query2ref
            result_path = Path(self.config['pose']['pi3']['results_dir']+"最相似") / f"{query_img.stem}.txt"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(result_path, final_pose)
############################################################简单求平均################################################
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
            result_path = Path(self.config['pose']['pi3']['results_dir']+"简单平均") / f"{query_img.stem}.txt"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(result_path, final_pose)


        
############################################################运动平均##################################################

            ma = MotionAveraging()
            q2r_poses = [np.linalg.inv(P_ref) @ P_query for P_ref in P_refs]
            # r2q_poses = [np.linalg.inv(P_query) @ P_ref for P_ref in P_refs]
            for q2r_pose in q2r_poses:
                q2r_pose[0:3,3] = q2r_pose[0:3,3] / np.linalg.norm(q2r_pose[0:3,3])
            final_pose = ma.motion_averaging(ref_poses_gt, q2r_poses)
        else:            
            if query_depth is not None:
                if Path(query_depth).exists():
                    logging.info(f"query_depth 存在")
                    depth_map = torch.norm(results['local_points'][0][0], dim=-1).cpu().numpy()  # [H, W]
                    query_depth = np.load(query_depth)
                    query_depth = cv2.resize(query_depth, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    scale_factor = compute_scale_factor(depth_map, query_depth,mask=query_mask)
            else:
                # 读取第一张ref图像
                ref_depth = None
                ref_img = Path(ref_imgs[0])
                if Path(ref_img.parent.parent/"depth"/f"{ref_img.stem}.npy").exists():
                    ref_depth = np.load(ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy")
                if Path(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy").exists():
                    ref_depth = np.load(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy")
                if ref_depth is not None:
                    depth_map = torch.norm(results['local_points'][0][1], dim=-1).cpu().numpy()  # [H, W]
                    ref_depth = cv2.resize(ref_depth, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    scale_factor = compute_scale_factor(depth_map, ref_depth,mask=ref_mask)
            if scale_factor is None:
                logging.error(f"error : scale_factor: {scale_factor}")
                scale_factor = 1.0
            query2ref = np.linalg.inv(P_refs[0]) @ P_query
            query2ref[:3, 3] *= scale_factor
            logging.info(f"scale_factor: {scale_factor}")
            # 初步计算最终的位姿
            final_pose = ref_poses_gt[0] @ query2ref

        result_path = Path(self.config['pose']['pi3']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(result_path, final_pose)
        np.savetxt(result_path.parent.parent/ f"last_pose.txt", final_pose)
        logging.info(f"pi3_final_pose: {final_pose}")
        depth = None
        map = None
        return final_pose ,depth,map

