from flask import Flask, request, jsonify
from pathlib import Path
import os
import numpy as np
from vps.core import VisualPositioningSystem
import logging
import cv2
import sys
import datetime
import json

app = Flask(__name__)

# Global VPS instance
vps = None

def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    global vps
    vps = VisualPositioningSystem(config_path)
    # Configure logging
    # Initialize VPS system
    logging.info("VPS system initialized successfully")
    
    return app

def get_robot_to_camera_transform(offset_z: float = -0.1):
    """
    构造一个从机器人中心坐标系到相机坐标系的变换矩阵（只有平移，没有旋转）
    offset_z: 相机坐标系下机器人在Z轴方向上的偏移,默认 -0.2m
    """
    T = np.eye(4)
    T[2, 3] = offset_z  
    return T

def transform_matrix_to_pose_2d(transform_matrix):
    """
    将4x4转换矩阵转换为2D位姿 (x, y, theta)
    
    Args:
        transform_matrix (np.ndarray): 4x4转换矩阵
        
    Returns:
        dict: 包含x, y, theta的字典
    """
    if transform_matrix is None:
        return None
    
    transform_matrix =  transform_matrix @ get_robot_to_camera_transform() 
    # 提取平移部分 (x, y, z)
    translation = transform_matrix[:3, 3]
    x, y, z = translation[0], translation[1], translation[2]
    
    # 提取旋转矩阵 (3x3)
    rotation_matrix = transform_matrix[:3, :3]
    
    # 从旋转矩阵计算yaw角 (绕z轴的旋转)
    # 使用atan2(R[1,0], R[0,0])来计算yaw角
    theta = np.arctan2(rotation_matrix[0, 0], -1*rotation_matrix[1, 0])
    
    # 转换为度数
    theta_degrees = np.degrees(theta)
    
    return {
        'x': float(x),
        'y': float(y),
        # 'z': float(z),
        # 'theta': float(theta_degrees),  # 以度为单位
        'theta': float(theta),      # 以弧度为单位
        # 'z': float(z)                   # 保留z值供参考
    }

def map2map(x, y, theta):
    scale = 1
    R = np.array([[ -0.20886882 ,0.97588923 ],[ -0.97588923 ,-0.20886882]])
    t = np.array([ 6.36628206, 9.91735557])
    src_pos = np.array([x, y])
    tgt_pos = scale * (R @ src_pos) + t

    dir_vec = np.array([np.cos(theta), np.sin(theta)])  
    new_dir = R @ dir_vec
    new_theta = np.arctan2(new_dir[1], new_dir[0])

    return tgt_pos[0], tgt_pos[1], new_theta

@app.route('/localize', methods=['POST'])
def localize():
    """Localization endpoint."""
    global vps
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        rgb_file = request.files['image']
        logging.info(f"rgb_file: {rgb_file.filename}")
        if rgb_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        # Check file format
        allowed_extensions = vps.config['service']['supported_formats']
        if not any(rgb_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': f'Unsupported file format. Supported: {allowed_extensions}'}), 400
        # Create temp directory if it doesn't exist
        os.makedirs(vps.config['service']['temp_dir'], exist_ok=True)
        query_image_path = os.path.join(vps.config['service']['temp_dir'], vps.config['service']['temp_rgb_name'])
        rgb_file.save(query_image_path)
        ##
        ##
        # 处理可选的深度文件
        depth_path = None
        if 'depth' in request.files:
            depth_file = request.files['depth']
            
            if depth_file and depth_file.filename:
                logging.info(f"接收到深度文件: {depth_file.filename}")
                file_ext = os.path.splitext(depth_file.filename)[1].lower()
                depth_path = os.path.join(vps.config['service']['temp_dir'], vps.config['service']['temp_depth_name'])

                if file_ext == '.png':
                    # 从文件流读取、解码、转换并保存
                    filestr = depth_file.read()
                    npimg = np.frombuffer(filestr, np.uint8)
                    depth_data_png = cv2.imdecode(npimg, cv2.IMREAD_ANYDEPTH)
                    if depth_data_png is not None:
                        depth_data = depth_data_png.astype(np.float32) / 1000.0
                        np.save(depth_path, depth_data)
                        logging.info(f"PNG深度文件已转换为NPY并保存到: {depth_path}")

                elif file_ext == '.npy':
                    # 直接保存npy文件
                    depth_file.save(depth_path)
                    logging.info(f"NPY深度文件已保存到: {depth_path}")
        ##
        ##
        # 指定大致pose范围
        last_pose = None
        if 'last_pose' in request.files:
            last_pose_file = request.files['last_pose']
            last_pose_json = json.load(last_pose_file)
            last_pose = np.array([last_pose_json['x'], last_pose_json['y'], last_pose_json['z']])
        if 'last_pose' in request.form:
            last_pose_json = json.loads(request.form['last_pose'])
            last_pose = np.array([last_pose_json['x'], last_pose_json['y'], last_pose_json['z']])
        if last_pose is not None:
            logging.info(f"client set last_pose: {last_pose}")
        # 执行定位 (depth_path可能是None,last_pose可能是None)
        pose3d= vps.localize(query_image_path,depth_path, last_pose)
        
        # Convert numpy arrays to lists for JSON serialization
        if pose3d is not None:
            # 如果result包含pose字段且是4x4矩阵
            # 将4x4矩阵转换为x, y, theta格式
            pose_2d = transform_matrix_to_pose_2d(pose3d)
            ans = map2map(x= pose_2d['x'], y= pose_2d['y'], theta= pose_2d['theta'])
            # ans = pose_2d
            ans = {
                'x': ans[0],
                'y': ans[1],
                'theta': ans[2]
            }
            logging.info(f"ans: {ans}")
            return jsonify(ans), 200
        else:
            return jsonify({'error': 'No pose found'}), 400
    except Exception as e:
        logging.error(f"Error during localization: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('./log',exist_ok=True)
    log_filename = datetime.datetime.now().strftime("log/log_%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    # 全局日志配置
    # logging.basicConfig(
    #     level=logging.INFO,  # 设置最低显示级别为 INFO
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 格式
    #     handlers=[
    #         # logging.FileHandler("log.txt"),       # 写入文件 log.txt
    #         logging.StreamHandler(sys.stdout)     # 同时输出到控制台
    #     ]
    # )
    # logger = logging.getLogger(__name__)
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False) 
    