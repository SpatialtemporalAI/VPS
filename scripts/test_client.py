#!/usr/bin/env python3
"""
VPS Test Client

This script demonstrates how to use the VPS service.
"""

import requests
import json
from pathlib import Path
import argparse
import os
import cv2
import numpy as np

def test_localization(base_url: str, image_path: str, depth_path: str):
    """Test the localization endpoint."""
    print(f"Testing localization with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        # 如果depth文件存在，则添加到files中
        depth_file = None
        if depth_path and os.path.exists(depth_path):
            depth_file = open(depth_path, 'rb')
            files['depth'] = depth_file
        data = {'last_pose': json.dumps({'x': 0, 'y': 0, 'z': 0})}
        try:
            response = requests.post(f"{base_url}/localize", files=files, data=data)
        finally:
            # 确保depth文件被关闭
            if depth_file:
                depth_file.close()
        
        if response.status_code == 200:
            result = response.json()
                        # 1. 处理姿态信息
            received_pose = result.get('pose')

            if received_pose and received_pose.get('x') is not None:
                print("\n--- 姿态信息 (Pose) ---")
                print(f"X: {received_pose['x']:.3f}")
                print(f"Y: {received_pose['y']:.3f}")
                print(f"Theta: {received_pose['theta']:.3f}")
            else:
                print("未接收到有效的姿态信息。")


            # 2. 重构地图 (NumPy 数组)
            map_data = result.get('map_data')
            map_shape = result.get('map_shape')
            map_dtype = result.get('map_dtype')

            if map_data and map_shape and map_dtype:
                print("\n--- 地图信息 (Map) ---")
                try:
                    # 将嵌套列表转换为 NumPy 数组
                    # 注意：使用 tolist() 转换为列表后，数组的形状信息通常在列表中已经体现
                    reconstructed_map = np.array(map_data, dtype=map_dtype)
                    
                    # 验证重构后的数组形状
                    if list(reconstructed_map.shape) != map_shape:
                        # 如果数组是从列表创建的，形状应该已经匹配，但保险起见可以检查
                        # 如果数据不是标准的二维或三维，可能需要 reshape
                        # reconstructed_map = reconstructed_map.reshape(map_shape) 
                        pass
                        
                    print(f"✅ 地图重构成功！形状: {reconstructed_map.shape}, 类型: {reconstructed_map.dtype}")
                    
                    # 示例：显示地图的部分内容或统计信息
                    print(f"地图像素值 Min/Max: {np.min(reconstructed_map)} / {np.max(reconstructed_map)}")
                    
                    # 接下来，您就可以像使用服务器上的 'map' 变量一样使用 'reconstructed_map' 了！
                    # 例如：
                    cv2.imwrite("Reconstructed.png", reconstructed_map)
                    
                except Exception as e:
                    print(f"❌ 地图重构失败: {e}")
                    print("请检查 map_data 是否为正确的嵌套列表格式。")
            else:
                print("未接收到完整的地图数据（缺少 map_data, map_shape, 或 map_dtype）。")

        else:
            print(f"✗ Localization failed: {response.status_code}")
            print(f"Error: {response}")

def main():
    parser = argparse.ArgumentParser(description='Test VPS service')
    parser.add_argument('--url', type=str, default='http://10.16.242.37:5000',
                       help='Base URL of the VPS service')
    parser.add_argument('--image', type=str, required=True,default="/home/phw/visual-localization/VPS/data/ref/rgb/frame_000000.jpg",
                       help='Path to test image')
    parser.add_argument('--depth', type=str, required=False,default=None,
                       help='Path to test image depth')

    args = parser.parse_args()
    
    # Test health check
    # test_health_check(args.url)
    # print()
    
    # Test localization
    test_localization(args.url, args.image, args.depth)

if __name__ == '__main__':
    main() 