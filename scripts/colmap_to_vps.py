#!/usr/bin/env python3
"""
COLMAP to VPS Data Converter

This script converts COLMAP format data to VPS reference database format.
Step 1: Convert basic data (RGB, poses, calibration)
Step 2: Generate render data from mesh (optional)
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import shutil
import re
import argparse
import json
import time
import yaml
import cv2

##todo: 添加原始depth

class COLMAPToVPSConverter:
    def __init__(self, colmap_dir: Path, output_dir: Path):
        """
        Initialize the converter for basic data conversion.
        
        Args:
            colmap_dir: Path to COLMAP directory containing images/ and sparse/
            output_dir: Path to output VPS reference database
        """
        self.colmap_dir = Path(colmap_dir)
        self.output_dir = Path(output_dir)
        
        # COLMAP paths
        self.images_dir = self.colmap_dir / "images"
        self.sparse_dir = self.colmap_dir / "sparse"
        self.raw_depth_dir = self.colmap_dir / "depth"
        self.cameras_txt = self.sparse_dir / "cameras.txt"
        self.images_txt = self.sparse_dir / "images.txt"
        # VPS output paths
        self.rgb_dir = self.output_dir / "rgb"
        self.poses_dir = self.output_dir / "poses"
        self.calibration_dir = self.output_dir / "calibration"
        self.depth_dir = self.output_dir / "depth"
        
        # Create output directories
        for dir_path in [self.rgb_dir, self.poses_dir, self.calibration_dir, self.depth_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Camera intrinsics cache
        self.cam_intrinsics = {}

    def _load_camera_intrinsics(self):
        """Load camera intrinsics from COLMAP cameras.txt."""
        print("Loading camera intrinsics...")
        with open(self.cameras_txt, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                cam_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
                self.cam_intrinsics[cam_id] = {
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'width': width, 'height': height
                }
        print(f"✓ Loaded {len(self.cam_intrinsics)} camera intrinsics")

    def _process_image_pose(self, img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name):
        """Process a single image pose and generate basic files."""
        # Copy RGB image
        src_img_path = self.images_dir / img_name
        dst_img_path = self.rgb_dir / img_name
        shutil.copyfile(src_img_path, dst_img_path)
        name = Path(img_name).stem
        for ext in ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]:
            src_depth_path = Path(self.raw_depth_dir) / f"{name}.{ext}"
            dst_depth_path = Path(self.depth_dir) / f"{name}.npy"
            if src_depth_path.exists():
                ref_depth = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
                ref_depth = ref_depth.astype(np.float32) /1000.0
                np.save(dst_depth_path, ref_depth)

        # Calculate camera-to-world pose
        quat = [qx, qy, qz, qw]
        t = np.array([tx, ty, tz])
        R_c2w = Rotation.from_quat(quat).as_matrix().T
        t_c2w = -R_c2w @ t
        T = np.eye(4)
        T[:3, :3] = R_c2w
        T[:3, 3] = t_c2w
        
        # Save pose as 4x4 matrix (txt format)
        pose_path = self.poses_dir / f"{Path(img_name).stem}.txt"
        np.savetxt(pose_path, T, fmt='%.8f', delimiter='\t')
        
        # Save calibration as 3x3 matrix (txt format)
        fx, fy, cx, cy = (self.cam_intrinsics[cam_id]['fx'], 
                         self.cam_intrinsics[cam_id]['fy'],
                         self.cam_intrinsics[cam_id]['cx'], 
                         self.cam_intrinsics[cam_id]['cy'])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        calib_path = self.calibration_dir / f"{Path(img_name).stem}.txt"
        np.savetxt(calib_path, K, fmt='%.6f', delimiter=' ')

    def convert_basic_data(self):
        """Convert COLMAP data to basic VPS format (RGB, poses, calibration)."""
        print(f"Converting COLMAP data from: {self.colmap_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load camera intrinsics
        self._load_camera_intrinsics()
        
        # Process images and poses
        print("Processing images and poses...")
        image_exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')
        with open(self.images_txt, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        # 统计所有图片行的数量作为total_images
        total_images = sum(1 for line in lines if len(line.split()) > 0 and line.split()[-1].lower().endswith(image_exts))
        processed_count = 0 
        
        for line in lines:
            parts = line.split()
            if len(parts) > 0 and parts[-1].lower().endswith(image_exts):
                # 这是图像信息行，处理即可
                img_id, qw, qx, qy, qz = map(float, parts[0:5])
                tx, ty, tz = map(float, parts[5:8])
                cam_id = int(parts[8])
                img_name = parts[9]
                
                # Process the image
                self._process_image_pose(img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{total_images} images...")
        
        print(f"✓ Basic data conversion complete!")
        print(f"  - Total images processed: {processed_count}")
        print(f"  - Output directory: {self.output_dir}")
        
        # Create summary
        self._create_summary()

    def _create_summary(self):
        """Create a summary of the converted data."""
        summary = {
            "conversion_info": {
                "source": str(self.colmap_dir),
                "output": str(self.output_dir),
                "timestamp": time.time()
            },
            "data_counts": {
                "rgb_images": len(list(self.rgb_dir.glob("*.jpg"))) + len(list(self.rgb_dir.glob("*.png"))),
                "poses": len(list(self.poses_dir.glob("*.txt"))),
                "calibrations": len(list(self.calibration_dir.glob("*.txt")))
            },
            "camera_intrinsics": self.cam_intrinsics
        }
        
        summary_path = self.output_dir / "colmap2data_conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP data to VPS basic format')
    parser.add_argument('--config', type=str, default='configs/data.yaml')
    parser.add_argument('--colmap_dir', type=str, default=None,
                       help='Path to COLMAP directory (containing images/ and sparse/)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to output VPS reference database')
    
    args = parser.parse_args()
    # 读取配置文件
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # 命令行参数优先生效
    colmap_dir = Path(args.colmap_dir) if args.colmap_dir else Path(cfg['colmap_dir'])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg['output_dir'])
    
    # Validate inputs
    if not colmap_dir.exists():
        print(f"❌ COLMAP directory not found: {colmap_dir}")
        return
    
    if not (colmap_dir / "images").exists() or not (colmap_dir / "sparse").exists():
        print(f"❌ Invalid COLMAP directory structure. Expected images/ and sparse/ subdirectories.")
        return
    
    # Create converter and run conversion
    converter = COLMAPToVPSConverter(
        colmap_dir=colmap_dir,
        output_dir=output_dir
    )
    
    converter.convert_basic_data()

if __name__ == "__main__":
    main() 