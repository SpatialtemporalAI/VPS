#!/usr/bin/env python3
"""
Test script for creating half-by-half diagonal split video
"""

from pathlib import Path
from create_half_by_half_video import create_half_by_half_video

def main():
    # 配置路径
    folder_a = Path("outputs/frames724")  # 左下角图像文件夹
    folder_b = Path("/home/phw/visual-localization/PGSR/office-07-01/train/ours_15000/renders")  # 右上角图像文件夹
    output_path = Path("outputs/pi3_depthgt_724_ma.mp4")  # 输出视频路径
    
    # 检查文件夹是否存在
    if not folder_a.exists():
        print(f"❌ 文件夹A不存在: {folder_a}")
        return
    
    if not folder_b.exists():
        print(f"❌ 文件夹B不存在: {folder_b}")
        return
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("开始创建对角线分割视频...")
    print(f"文件夹A (左下角): {folder_a}")
    print(f"文件夹B (右上角): {folder_b}")
    print(f"输出视频: {output_path}")
    
    # 创建视频
    success = create_half_by_half_video(
        folder_a=folder_a,
        folder_b=folder_b,
        output_path=output_path,
        fps=3,  # 每秒3帧
        target_size=(1280, 960),  # 总尺寸
        add_labels=True,
        label_a="Query",  # 左下角标签
        label_b="Result in mesh"  # 右上角标签
    )
    
    if success:
        print("🎉 对角线分割视频创建成功!")
        print(f"视频文件: {output_path}")
    else:
        print("❌ 视频创建失败!")

if __name__ == "__main__":
    main() 