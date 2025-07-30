#!/usr/bin/env python3
"""
处理渲染数据脚本
将三个文件夹中的渲染图像、深度和姿态数据进行处理：
1. RGB图像复制到指定位置
2. 深度从jpg毫米单位转换为npy米单位
3. 只处理三个文件夹中都存在的文件
"""

import os
import shutil
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_common_files(rgb_dir, depth_dir, pose_dir):
    """
    获取三个文件夹中都存在的文件（除了后缀名）
    
    Args:
        rgb_dir: RGB图像文件夹路径
        depth_dir: 深度图像文件夹路径
        pose_dir: 姿态文件文件夹路径
    
    Returns:
        list: 共同文件名列表（不含后缀）
    """
    # 获取各文件夹中的文件名（不含后缀）
    rgb_files = {Path(f).stem for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    depth_files = {Path(f).stem for f in os.listdir(depth_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    pose_files = {Path(f).stem for f in os.listdir(pose_dir) if f.lower().endswith('.txt')}
    
    # 找到三个文件夹中都存在的文件
    common_files = rgb_files & depth_files & pose_files
    
    logger.info(f"RGB文件夹文件数: {len(rgb_files)}")
    logger.info(f"深度文件夹文件数: {len(depth_files)}")
    logger.info(f"姿态文件夹文件数: {len(pose_files)}")
    logger.info(f"共同文件数: {len(common_files)}")
    
    return sorted(list(common_files))

def find_file_with_stem(directory, stem, extensions):
    """
    在指定目录中查找具有特定stem和扩展名的文件
    
    Args:
        directory: 目录路径
        stem: 文件名（不含后缀）
        extensions: 可能的扩展名列表
    
    Returns:
        str: 找到的文件完整路径,如果没找到返回None
    """
    for ext in extensions:
        file_path = os.path.join(directory, f"{stem}{ext}")
        if os.path.exists(file_path):
            return file_path
    return None

def convert_depth_jpg_to_npy(depth_jpg_path, output_npy_path):
    """
    将深度jpg图像从毫米单位转换为npy米单位
    
    Args:
        depth_jpg_path: 深度jpg文件路径
        output_npy_path: 输出npy文件路径
    """
    try:
        # 读取深度图像
        depth_img = Image.open(depth_jpg_path)
        depth_array = np.array(depth_img)
        
        # 如果是RGB图像，转换为灰度
        if len(depth_array.shape) == 3:
            # 假设深度值存储在某个通道中，这里取第一个通道
            depth_array = depth_array[:, :, 0]
        
        # 从毫米转换为米 (除以1000)
        depth_meters = depth_array.astype(np.float32) / 1000.0
        
        # 保存为npy文件
        np.save(output_npy_path, depth_meters)
        
        logger.info(f"深度转换完成: {depth_jpg_path} -> {output_npy_path}")
        logger.info(f"深度范围: {depth_meters.min():.3f}m - {depth_meters.max():.3f}m")
        
    except Exception as e:
        logger.error(f"深度转换失败 {depth_jpg_path}: {str(e)}")

def process_data(rgb_dir, depth_dir, pose_dir, output_rgb_dir, output_depth_dir, output_pose_dir):
    """
    处理三个文件夹中的数据
    
    Args:
        rgb_dir: RGB图像输入文件夹
        depth_dir: 深度图像输入文件夹
        pose_dir: 姿态文件输入文件夹
        output_rgb_dir: RGB图像输出文件夹
        output_depth_dir: 深度npy文件输出文件夹
        output_pose_dir: 姿态文件输出文件夹
    """
    # 创建输出目录
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    os.makedirs(output_pose_dir, exist_ok=True)
    
    # 获取共同文件
    common_files = get_common_files(rgb_dir, depth_dir, pose_dir)
    
    if not common_files:
        logger.warning("没有找到三个文件夹中都存在的文件！")
        return
    
    processed_count = 0
    
    for file_stem in common_files:
        try:
            # 查找对应的文件
            rgb_file = find_file_with_stem(rgb_dir, file_stem, ['.jpg', '.jpeg', '.png'])
            depth_file = find_file_with_stem(depth_dir, file_stem, ['.jpg', '.jpeg', '.png'])
            pose_file = find_file_with_stem(pose_dir, file_stem, ['.txt'])
            
            if not all([rgb_file, depth_file, pose_file]):
                logger.warning(f"文件 {file_stem} 在某个文件夹中未找到，跳过")
                continue
            
            # 处理RGB图像 - 复制到输出目录
            rgb_output_path = os.path.join(output_rgb_dir, f"{file_stem}.jpg")
            if rgb_file:  # 确保文件路径不为None
                shutil.copy2(rgb_file, rgb_output_path)
            
            # 处理深度图像 - 转换为npy
            depth_output_path = os.path.join(output_depth_dir, f"{file_stem}.npy")
            if depth_file:  # 确保文件路径不为None
                convert_depth_jpg_to_npy(depth_file, depth_output_path)
            
            # 处理姿态文件 - 复制到输出目录
            pose_output_path = os.path.join(output_pose_dir, f"{file_stem}.txt")
            if pose_file:  # 确保文件路径不为None
                shutil.copy2(pose_file, pose_output_path)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"已处理 {processed_count} 个文件...")
                
        except Exception as e:
            logger.error(f"处理文件 {file_stem} 时出错: {str(e)}")
            continue
    
    logger.info(f"处理完成！共处理了 {processed_count} 个文件")
    logger.info(f"RGB图像输出到: {output_rgb_dir}")
    logger.info(f"深度npy文件输出到: {output_depth_dir}")
    logger.info(f"姿态文件输出到: {output_pose_dir}")

def main():
    parser = argparse.ArgumentParser(description='处理渲染数据：RGB、深度和姿态文件')
    parser.add_argument('--rgb_dir', type=str, required=True, help='RGB图像输入文件夹路径')
    parser.add_argument('--depth_dir', type=str, required=True, help='深度图像输入文件夹路径')
    parser.add_argument('--pose_dir', type=str, required=True, help='姿态文件输入文件夹路径')
    parser.add_argument('--output_rgb_dir', type=str, required=True, help='RGB图像输出文件夹路径')
    parser.add_argument('--output_depth_dir', type=str, required=True, help='深度npy文件输出文件夹路径')
    parser.add_argument('--output_pose_dir', type=str, required=True, help='姿态文件输出文件夹路径')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    for dir_path, dir_name in [
        (args.rgb_dir, "RGB图像"),
        (args.depth_dir, "深度图像"),
        (args.pose_dir, "姿态文件")
    ]:
        if not os.path.exists(dir_path):
            logger.error(f"{dir_name}目录不存在: {dir_path}")
            return
    
    logger.info("开始处理数据...")
    logger.info(f"RGB输入目录: {args.rgb_dir}")
    logger.info(f"深度输入目录: {args.depth_dir}")
    logger.info(f"姿态输入目录: {args.pose_dir}")
    
    process_data(
        args.rgb_dir,
        args.depth_dir,
        args.pose_dir,
        args.output_rgb_dir,
        args.output_depth_dir,
        args.output_pose_dir
    )

if __name__ == "__main__":
    main()
