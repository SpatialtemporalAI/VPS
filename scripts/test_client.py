#!/usr/bin/env python3
"""
VPS Test Client

This script demonstrates how to use the VPS service.
"""

import requests
import json
import argparse
import os
import time


def test_localization(base_url: str, image_path: str, depth_path: str, robot_id: str):
    """Test the localization endpoint."""
    print(f"Testing localization with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        # 如果depth文件存在，则添加到files中
        depth_file = None
        if depth_path and os.path.exists(depth_path):
            depth_file = open(depth_path, 'rb')
            files['depth'] = depth_file
        data = {
            'robot_id': robot_id,
            'last_pose': json.dumps({'x': 0, 'y': 0, 'z': 0}),
        }
        try:
            response = requests.post(f"{base_url}/localize_by_light", files=files, data=data)
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
            # map_data = result.get('map_png_b64')
            # map_shape = result.get('map_shape')
            # map_dtype = result.get('map_dtype')
            # img_bytes = base64.b64decode(result['map_png_b64'])
            # occ_map = cv2.imdecode(
            #     np.frombuffer(img_bytes, np.uint8),
            #     cv2.IMREAD_GRAYSCALE
            # )
            # print("\n--- 重构后的占用地图 (Occupancy Map) ---")
            # if occ_map is not None:
            #     print(f"✅ 地图重构成功！形状: {occ_map.shape}, 类型: {occ_map.dtype}")
            #     print(f"地图像素值 Min/Max: {np.min(occ_map)} / {np.max(occ_map)}")
            #     # 示例：保存重构的地图为 PNG 文件
            #     cv2.imwrite("Reconstructed_Occupancy_Map.png", occ_map)
            # else:
            #     print("未接收到完整的地图数据（缺少 map_data, map_shape, 或 map_dtype)。")

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
    parser.add_argument('--robot-id', type=str, default='test',
                       help='Robot ID sent to the VPS service')

    args = parser.parse_args()
    
    # Test health check
    # test_health_check(args.url)
    # print()
    
    # Test localization
    s = time.time()
    test_localization(args.url, args.image, args.depth, args.robot_id)
    e = time.time()
    print(f"Time taken: {e - s} seconds")

if __name__ == '__main__':
    main() 
