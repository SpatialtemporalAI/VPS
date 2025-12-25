from .vpr import VisualPlaceRecognition
import yaml
from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
class VisualPositioningSystem:
    """Main class for the Visual Positioning System."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the VPS system.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.vpr = VisualPlaceRecognition(self.config)
        
        # Initialize pose estimator based on method
        pose_method = self.config['pose']['method']
        if pose_method == 'vggt':
            from .pose_vggt import PoseEstimatorVGGT
            self.pose_estimator = PoseEstimatorVGGT(self.config)
        elif pose_method =='vggt_nav':
            from.pose_vggt_fixdepth import PoseEstimator
            self.pose_estimator = PoseEstimator(self.config)
        elif pose_method == 'mast3r':
            from .pose_mast3r import PoseEstimatorMASt3R
            self.pose_estimator = PoseEstimatorMASt3R(self.config)
        elif pose_method == 'pi3':
            from .pose_pi3 import PoseEstimatorPi3
            self.pose_estimator = PoseEstimatorPi3(self.config)
        else:
            raise ValueError(f"Unsupported pose method: {pose_method}")
        self.use_depth_pre = self.config['pose']['use_depth_pre']
        if self.use_depth_pre:
            from .depth_pred import DepthPred
            self.depth_model = DepthPred(self.config)

    def _vpr_task(self, query_image, last_pose):
        """VPR任务,在独立线程中执行"""
        start_time = time.time()
        similar_pairs = self.vpr.find_similar_images(query_image, last_pose)
        end_time = time.time()
        logging.info(f"VPR time: {end_time - start_time} seconds")
        return similar_pairs

    def _depth_task(self, query_image):
        """深度预测任务，在独立线程中执行"""
        start_time = time.time()
        out = self.depth_model.infer(query_image)
        end_time = time.time()
        logging.info(f"Depth prediction time: {end_time - start_time} seconds")
        return out

    def localize(self,
                 query_image: Union[str, Path],
                 query_depth: Optional[Union[str, Path]] = None,
                 last_pose: Optional[np.ndarray] = None
                ) -> np.ndarray:
        """
        Perform visual localization for a single query image.
        
        Args:
            query_image: Path to query image
            query_depth: Path to query depth map(npy)
        Returns:
            Dictionary containing:
            - pose: 4x4 transformation matrix c2w
            - processing_time: processing time in seconds
        """
        a = time.time()
        if self.use_depth_pre and query_depth is None:
            # 并行执行VPR和深度预测
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交VPR任务
                vpr_future = executor.submit(self._vpr_task, query_image, last_pose)
                
                # 提交深度预测任务
                depth_future = executor.submit(self._depth_task, query_image)
                # 等待VPR完成
                vpr_future.result()
                # 等待深度预测完成（如果有）
                depth_result = depth_future.result()
                # 保存预测的深度到临时文件
                depth_path = os.path.join(self.config['service']['temp_dir'], self.config['service']['temp_depth_name'])
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                np.save(depth_path, depth_result['depth'].cpu().numpy())
                query_depth = depth_path
            
                b = time.time()
                logging.info(f"Parallel VPR + Depth time: {b - a} seconds")
        else:
            self.vpr.find_similar_images(query_image, last_pose)
            b = time.time()
            logging.info(f"VPR time: {b - a} seconds")
        b = time.time()
        # 执行姿态估计
        pose_answer,depth,new_map = None,None,None
        pose_answer,depth,new_map = self.pose_estimator.estimate_pose(query_image, query_depth)
        # pose_answer = self.pose_estimator.estimate_pose(query_image, query_depth)
        c = time.time()
        logging.info(f"Pose estimation time: {c - b} seconds")
        logging.info(f"VPS total time: {c - a} seconds")
        return pose_answer,depth,new_map

