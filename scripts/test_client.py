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
            print((f"result: {result}"))

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