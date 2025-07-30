import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import cv2
import json
import logging
import sys
sys.path.append("third_party/mast3r")
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from scipy.spatial.transform import Rotation as R
from vps.utils.processing import compute_scale_factor, generate_ref_list



class PoseEstimatorMASt3R:
    """Pose estimation module using MASt3R."""
    
    def __init__(self, config: Dict):
        """
        Initialize the MASt3R pose estimator.
        
        Args:
            config: Configuration dictionary containing pose estimation settings
        """
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # Initialize MASt3R model
        model_path = config['pose']['mast3r']['model_path']
        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(self.device).eval()
        
        # MASt3R specific settings
        self.image_size = config['pose']['mast3r']['image_size']
        self.batch_size = config['pose']['mast3r']['batch_size']

    def run_MASt3R(self, ref_img: Union[str, Path], query_img: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MASt3R inference on a pair of images.
        
        Args:
            ref_img: Path to reference image
            query_img: Path to query image
            
        Returns:
            Tuple containing:
            - Relative pose matrix (4x4) query2ref
            - Depth map from MASt3R
        """
        # Load and preprocess images
        images = load_images([str(ref_img), str(query_img)], size=self.image_size)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        
        # Run inference
        output = inference(pairs, self.model, self.device, batch_size=self.batch_size)
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)
        
        # Get poses
        poses = scene.get_im_poses()
        
        # Determine relative pose (query2ref)
        P_ini = np.eye(4)
        flag = poses[0].detach().cpu().numpy()
        
        if np.allclose(flag, P_ini, atol=1e-6):
            P_rel = poses[1].detach().cpu().numpy()  # query2ref
        else:
            P_rel = poses[0].detach().cpu().numpy()
            P_rel = np.linalg.inv(P_rel)  # query2ref
        
        # Get depth map
        pred_depth_ref = scene.get_depthmaps()[0].cpu().numpy()
        pred_depth_query = scene.get_depthmaps()[0].cpu().numpy()
        return P_rel, pred_depth_ref, pred_depth_query

    def estimate_pose(
        self, 
        query_img: Union[str, Path],
        query_depth: Optional[Union[str, Path]] = None
        ) -> np.ndarray:
        """
        Estimate absolute pose for query image using MASt3R.
        
        Args:
            query_img: Path to query image
            query_depth: Optional Path to query depth
            
        Returns:
            Absolute pose matrix (4x4) camera2world
        """
        query_img = Path(query_img)
        ref_imgs = generate_ref_list(query_img, self.config['pose']['mast3r']['ref_dir'], self.config['vpr']['pairs_file_path'])
        
        if not ref_imgs:
            raise ValueError(f"No reference images found for query {query_img.name}")
        
        # Use the first reference image
        ref_img = Path(ref_imgs[0])
        
        # Run MASt3R to get relative pose and depth
        P_rel, pred_depth_ref, pred_depth_query = self.run_MASt3R(ref_img, query_img)
        scale_factor = 1.0
        if query_depth is not None:
            if Path(query_depth).exists():
                logging.info(f"query_depth 存在")
                query_depth = np.load(query_depth)
                query_depth = cv2.resize(query_depth, (pred_depth_query.shape[1], pred_depth_query.shape[0]), interpolation=cv2.INTER_LINEAR)
                scale_factor = compute_scale_factor(pred_depth_query, query_depth)
        else:
            # Try to load reference depth for scale recovery
            ref_depth = None
            depth_path = ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy"
            depth_render_path = ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy"
            if depth_path.exists():
                ref_depth = np.load(depth_path)
            elif depth_render_path.exists():
                ref_depth = np.load(depth_render_path)
            if ref_depth is not None:
                ref_depth = cv2.resize(ref_depth, 
                                            (pred_depth_ref.shape[1], pred_depth_ref.shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)
                scale_factor = compute_scale_factor(pred_depth_ref, ref_depth)
        P_rel[:3, 3] *= scale_factor
        logging.info(f"scale_factor: {scale_factor}")
        # Load reference pose and depth for scale recovery
        ref_pose = np.loadtxt(ref_img.parent.parent / "poses" / f"{ref_img.stem}.txt").reshape(4, 4)
        # Compute final absolute pose
        final_pose = ref_pose @ P_rel
        
        # Save result
        result_path = Path(self.config['pose']['mast3r']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)
        np.savetxt(result_path.parent.parent/ f"last_pose.txt", final_pose)
        logging.info(f"mast3r_final_pose: {final_pose}")
        return final_pose

