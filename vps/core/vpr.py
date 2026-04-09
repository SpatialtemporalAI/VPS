from re import S
import torch
from pathlib import Path
from typing import List, Dict, Union, Optional
import shutil
import numpy as np
import json
import time
import os
from hloc import extract_features, extractors
from hloc.utils.base_model import dynamic_load
from vps.utils.find_similar import find_similar, parse_names, get_descriptors, find_similar_vpr_pose
from hloc.utils.io import list_h5_names
import logging
class VisualPlaceRecognition:
    """Visual Place Recognition module using Hloc's methods with pose-based spatial filtering."""
    
    def __init__(self, config: Dict):
        """
        Initialize the VPR module.
        
        Args:
            config: Configuration dictionary containing VPR settings
        """
        self.config = config
        self.method = config['vpr']['method']
        self.top_k = config['vpr']['top_k']
        self.size_num_matched = config['vpr']['size_num_matched']
        self.ref_data_path = config['vpr']['ref_data_path']
        self.similarity_threshold = config['vpr'].get('similarity_threshold', 0.7)  # 相似度阈值

        # 基于pose的空间过滤参数
        self.use_spatial_filtering = config['vpr'].get('use_spatial_filtering', False)
        self.spatial_radius = config['vpr'].get('spatial_radius', 2.0)  # 空间搜索半径（米）
        self.history_file = Path(config['vpr'].get('history_file', 'data/outputs/last_pose.txt'))
        self.last_pose = None

        #加载vpr模型
        if self.method == 'netvlad':
            self.retrieval_conf = extract_features.confs['netvlad']
        elif self.method == 'megaloc':
            self.retrieval_conf = extract_features.confs['megaloc']
        else:
            raise ValueError(f"Unsupported VPR method: {self.method}")
        #megaloc vpr
        self.model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])(self.retrieval_conf["model"]).eval().to(config['system']['device'])
        
        #加载数据库数据
        self.db_names = None
        self.db_desc = None
        self.ref_descriptors = None
        # Check if reference descriptors already exist
        self.ref_descriptors = Path(self.config['vpr']['ref_descriptors_path'])
        ref_images = Path(self.ref_data_path) / "rgb"
        # Extract reference descriptors only if they don't exist or check_flag is True
        check_flag = self.config['vpr'].get('check_ref', True)
        if not self.ref_descriptors.exists() or check_flag:
            logging.info("Extracting reference descriptors...")
            self.extract_global_descriptors(ref_images, self.ref_descriptors)
        else:
            logging.info("Using existing reference descriptors...")

        # 统一预加载ref的特征向量
        self.ref_descriptors = [self.ref_descriptors]
        name2db = {n: i for i, p in enumerate(self.ref_descriptors) for n in list_h5_names(p)}
        db_names_h5 = list(name2db.keys())
        self.db_names = parse_names(prefix=None, names=None, names_all=db_names_h5)
        self.db_desc = get_descriptors(self.db_names, self.ref_descriptors, name2db)

        #处理ref的pose文件 以tensor矩阵保存，顺序与ref的特征顺序对齐
        ref_poses_dir = Path(self.ref_data_path) / "poses"
        ref_poses = []
        for name in self.db_names:
            pose = np.loadtxt(Path(ref_poses_dir) / f"{Path(name).stem}.txt").reshape(4,4)
            ref_poses.append(pose)
        self.ref_poses_tensor = torch.from_numpy(np.stack(ref_poses, 0)).float()

    def _load_pose_history(self):
        """加载历史pose信息"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return np.loadtxt(f)[:3, 3]
            except:
                logging.info(f"没有历史pose文件 {self.history_file}, 请先进行定位")
        return None

    def _delete_history_file(self):
        """删除历史pose文件"""
        if self.history_file.exists():
            os.remove(self.history_file)
            logging.info(f"删除历史pose文件 {self.history_file}")

    def extract_global_descriptors(self, 
                                 images: Union[str, Path, List[Union[str, Path]]], 
                                 output_path: Union[str, Path]) -> Path:
        """
        Extract global descriptors from images.
        
        Args:
            images: Path to image or list of image paths
            output_path: Path to save the descriptors
            
        Returns:
            Path to the saved descriptors
        """
        return extract_features.main(self.retrieval_conf, images,feature_path=output_path, overwrite=True, model=self.model)


    def find_similar_images(self,
                          query_image: Union[str, Path],
                          last_pose: Optional[np.ndarray] = None,
                          ) -> Path:
        """
        Complete pipeline to find similar images for a query image with spatial filtering.
        
        Args:
            query_image: Path to the query image
            last_pose: Optional pose of the query image for spatial filtering np.array: xyz
            
        Returns:
            Path to the pairs file containing similar images
        """
        # Ensure ref_images is a list
        ref_images = Path(self.ref_data_path) / "rgb"
        if isinstance(ref_images, (str, Path)):
            ref_images = Path(ref_images)
        
        output_dir = Path(self.config['vpr']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create unified query image with fixed name (overwrite mode)
        unified_query_dir = output_dir / "temp"
        logging.info("Cleaning up query directory...")
        if unified_query_dir.exists():
            shutil.rmtree(unified_query_dir)
            logging.info(f"✓ Cleaned up: {unified_query_dir}")
        unified_query_dir.mkdir(exist_ok=True, parents=True)

        # 直接以原文件名复制到目标目录，会覆盖同名文件
        shutil.copy2(query_image, unified_query_dir / Path(query_image).name)
        
        # Extract query descriptors (always overwrite due to unified naming)
        logging.info("Extracting query descriptors...")
        start_time = time.time()
        query_descriptors = self.extract_global_descriptors(unified_query_dir, Path(self.config['vpr']['query_descriptors_path']))
        end_time = time.time()
        logging.info(f"query_image: {Path(query_image).name} extract_global_descriptors time: {end_time - start_time:.8f} seconds")
       
        if last_pose is None:
            self.last_pose = self._load_pose_history()
        else:
            self.last_pose = last_pose
        self._delete_history_file()
        start_time = time.time()
        if self.size_num_matched == 1:
            find_similar(
                query_descriptors=query_descriptors, #query 特征   Path
                db_descriptors=self.ref_descriptors, #refs 特征路径  list[path]
                db_names=self.db_names, #refs 名称 list
                db_desc=self.db_desc, # db map {name: descriptor}  torch.Tensor [N, D]
                output=self.config['vpr']['pairs_file_path'], # 输出文件路径
                num_matched=self.top_k, # 匹配数量 int
                similarity_threshold=self.similarity_threshold, # 相似度阈值 float
                last_pose=self.last_pose, # 历史pose np.ndarray [3,]
                spatial_radius=self.spatial_radius, # 空间搜索半径 float
                use_spatial_filtering=self.use_spatial_filtering, # 是否开启空间过滤 bool
                ref_poses_tensor=self.ref_poses_tensor # ref pose tensor torch.Tensor [N, 3]
            )
        else:

            find_similar_vpr_pose(
                query_name=Path(query_image).name,
                query_descriptors=query_descriptors, #query 特征   Path
                db_descriptors=self.ref_descriptors, #refs 特征路径  list[path]
                db_names=self.db_names, #refs 名称 list
                db_desc=self.db_desc, # db map {name: descriptor}  torch.Tensor [N, D]
                output=self.config['vpr']['pairs_file_path'], # 输出文件路径
                num_matched=self.top_k, # 匹配数量 int
                size_num_matched=self.size_num_matched, #几倍num_matched查找
                ref_poses_tensor=self.ref_poses_tensor # ref pose tensor torch.Tensor [N, 3]
                )   


        # find_similar_vpr_pose_kmeans(
        #     query_name=Path(query_image).name,
        #     query_descriptors=query_descriptors, #query 特征   Path
        #     db_descriptors=self.ref_descriptors, #refs 特征路径  list[path]
        #     db_names=self.db_names, #refs 名称 list
        #     db_desc=self.db_desc, # db map {name: descriptor}  torch.Tensor [N, D]
        #     output=self.config['vpr']['pairs_file_path'], # 输出文件路径
        #     num_matched=self.top_k, # 匹配数量 int
        #     ref_poses_tensor=self.ref_poses_tensor # ref pose tensor torch.Tensor [N, 3]
        #     )  
        end_time = time.time()
        logging.info(f"query_image: {Path(query_image).name} pairs_from_retrieval time: {end_time - start_time:.8f} seconds")
        
        return self.config['vpr']['pairs_file_path']