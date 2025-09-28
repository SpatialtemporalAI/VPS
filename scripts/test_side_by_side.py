#!/usr/bin/env python3
"""
Test script for creating side-by-side video
"""

from pathlib import Path
from create_side_by_side_video import create_side_by_side_video

def main():
    # 配置路径
    folder_a = Path("data/frames")  # 左半边图像文件夹
    folder_b = Path("/home/phw/visual-localization/PGSR/wanke/train/ours_15000/renders")  # 右半边图像文件夹
    output_path = Path("outputs/wanke_side_by_side.mp4")  # 输出视频路径
    
    # 检查文件夹是否存在
    if not folder_a.exists():
        print(f"❌ 文件夹A不存在: {folder_a}")
        return
    
    if not folder_b.exists():
        print(f"❌ 文件夹B不存在: {folder_b}")
        return
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("开始创建左右分屏视频...")
    print(f"文件夹A: {folder_a}")
    print(f"文件夹B: {folder_b}")
    print(f"输出视频: {output_path}")
    
    # 创建视频
    success = create_side_by_side_video(
        folder_a=folder_a,
        folder_b=folder_b,
        output_path=output_path,
        fps=3,  # 每秒10帧
        target_size=(848, 480),  # 每个半边640x480
        add_labels=True,
        label_a="Query",  # 左半边标签
        label_b="Result in mesh"  # 右半边标签
    )
    
    if success:
        print("🎉 视频创建成功!")
        print(f"视频文件: {output_path}")
    else:
        print("❌ 视频创建失败!")

if __name__ == "__main__":
    main() 