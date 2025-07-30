import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (qw, qx, qy, qz)
    
    Args:
        R: 3x3旋转矩阵
        
    Returns:
        tuple: (qw, qx, qy, qz)
    """
    # 使用scipy的Rotation类进行转换
    r = Rotation.from_matrix(R)
    quat = r.as_quat()  # 返回 [qx, qy, qz, qw]
    # COLMAP格式需要 [qw, qx, qy, qz]
    return quat[3], quat[0], quat[1], quat[2]

def c2w_to_w2c(c2w_matrix):
    """
    将camera-to-world矩阵转换为world-to-camera矩阵
    
    Args:
        c2w_matrix: 4x4的camera-to-world变换矩阵
        
    Returns:
        tuple: (rotation_matrix, translation_vector) 对应w2c
    """
    # w2c = inv(c2w)
    w2c_matrix = np.linalg.inv(c2w_matrix)
    
    # 提取旋转和平移
    rotation = w2c_matrix[:3, :3]
    translation = w2c_matrix[:3, 3]
    
    return rotation, translation

def convert_poses_to_colmap(poses_dir, output_file, image_extension='.jpg'):
    """
    将poses文件夹中的所有位姿文件转换为COLMAP格式
    
    Args:
        poses_dir: 包含pose txt文件的目录
        output_file: 输出的COLMAP images.txt文件路径
        image_extension: 图像文件的扩展名
    """
    poses_dir = Path(poses_dir)
    
    # 收集所有pose文件
    pose_files = sorted([f for f in poses_dir.glob('*.txt')])
    
    if not pose_files:
        print(f"在目录 {poses_dir} 中没有找到任何txt文件")
        return
    
    # 开始写入COLMAP格式文件
    with open(output_file, 'w') as f:
        # 写入头部注释
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(pose_files)}\n")
        
        for image_id, pose_file in enumerate(pose_files, 1):
            try:
                # 读取4x4位姿矩阵
                c2w_matrix = np.loadtxt(pose_file)
                
                if c2w_matrix.shape != (4, 4):
                    print(f"警告: {pose_file} 不是4x4矩阵,跳过")
                    continue
                
                # 转换为w2c
                rotation, translation = c2w_to_w2c(c2w_matrix)
                
                # 转换旋转矩阵为四元数
                qw, qx, qy, qz = rotation_matrix_to_quaternion(rotation)
                tx, ty, tz = translation
                
                # 构造图像文件名
                image_name = pose_file.stem + image_extension
                
                # 写入COLMAP格式的行
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                       f"{tx:.6f} {ty:.6f} {tz:.6f} 1 {image_name}\n")
                
                # COLMAP格式需要第二行为空（没有2D特征点）
                f.write("\n")
                
            except Exception as e:
                print(f"处理文件 {pose_file} 时出错: {e}")
                continue
    
    print(f"转换完成！输出文件: {output_file}")
    print(f"共处理了 {len(pose_files)} 个位姿文件")

if __name__ == '__main__':
    # --- 配置 ---
    poses_directory = Path("/home/phw/visual-localization/VPS/data/outputs/poses")  # poses文件夹路径
    output_colmap_file = Path("/home/phw/visual-localization/VPS/outputs/images_colmap.txt")  # 输出文件路径
    image_ext = '.jpg'  # 图像文件扩展名，根据你的实际情况修改（.jpg 或 .png）
    # ------------
    
    if not poses_directory.exists():
        print(f"错误: poses目录不存在 {poses_directory}")
    else:
        convert_poses_to_colmap(poses_directory, output_colmap_file, image_ext)
