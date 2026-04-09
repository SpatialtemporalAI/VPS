import os
import shutil  # 👈 新增：用于复制文件
import numpy as np
from scipy.spatial.transform import Rotation as R

def analyze_pose_degeneracy(poses, ref_ids):
    """
    分析10个Reference Pose的分布和朝向，判断是否存在共线或朝向过于一致的退化情况
    """
    print(f"\n{'='*20} 📊 Pose 诊断分析 ({len(poses)}个相机) {'='*20}")
    
    # 提取平移 (XYZ) 和 旋转矩阵 (3x3)
    translations = np.array([pose[0:3, 3] for pose in poses])
    rotations = np.array([pose[0:3, 0:3] for pose in poses])
    
    # ---------------- 1. 位置分布分析 (判断是否共线) ----------------
    print("\n[位置分布分析 (判断是否共线)]")
    std_xyz = np.std(translations, axis=0)
    print(f"  • X轴 标准差: {std_xyz[0]:.4f} m (侧向分布宽度)")
    print(f"  • Y轴 标准差: {std_xyz[1]:.4f} m (高度/前后分布宽度)")
    print(f"  • Z轴 标准差: {std_xyz[2]:.4f} m (前后/高度分布宽度)")
    
    # 使用 SVD 分析点云的“扁平”程度 (判断共线)
    # 如果最小奇异值极小，说明点都排在一条线上
    centroid = np.mean(translations, axis=0)
    centered_t = translations - centroid
    U, S, Vt = np.linalg.svd(centered_t)
    print(f"  • 空间分布奇异值 S: [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}]")
    if S[2] < 1e-2 or (S[0] / (S[2] + 1e-6)) > 50:
        print("  ⚠️ 警告: 平移点云呈现极度的【线状或平面退化】！(大家几乎排成一条线)")
    else:
        print("  ✅ 位置分布良好，具有 3D 体积感。")

    # ---------------- 2. 朝向一致性分析 (判断是否平行) ----------------
    print("\n[朝向一致性分析 (判断射线是否平行)]")
    # 假设相机的光轴(正前方)是旋转矩阵的 Z 轴 (第3列) -> [0, 0, 1]^T 乘过去
    forward_vectors = rotations[:, 0:3, 2] 
    
    angles = []
    for i in range(len(forward_vectors)):
        for j in range(i + 1, len(forward_vectors)):
            # 计算两个光轴向量的夹角
            cos_theta = np.clip(np.dot(forward_vectors[i], forward_vectors[j]), -1.0, 1.0)
            angle_deg = np.arccos(cos_theta) * 180.0 / np.pi
            angles.append(angle_deg)
            
    mean_angle = np.mean(angles)
    max_angle = np.max(angles)
    print(f"  • 两两光轴平均夹角: {mean_angle:.2f} 度")
    print(f"  • 两两光轴最大夹角: {max_angle:.2f} 度")
    
    if max_angle < 5.0:
        print("  ⚠️ 警告: 所有相机的朝向极其相似！(射线几乎平行，必定触发杠杆放大效应)")
    else:
        print("  ✅ 朝向具有一定视差。")
    print("="*60 + "\n")


def perturb_and_save_poses(poses, ref_ids, rgb_dir, output_dir="poses_perturbed", max_trans_m=0.20, max_rot_deg=15.0): # 👈 新增 rgb_dir 参数
    """
    为每个 pose 注入随机扰动并保存，同时复制对应的 RGB 图像
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始生成加噪后的 Poses 并复制图像，保存至文件夹: {output_dir}/")
    
    success_count = 0
    
    for pose, ref_id in zip(poses, ref_ids):
        new_pose = pose.copy()
        
        # --- 1. 制造随机平移噪声 ---
        trans_direction = (np.random.rand(3) - 0.5) * 2
        trans_direction /= np.linalg.norm(trans_direction)
        trans_magnitude = np.random.uniform(0, max_trans_m)
        trans_noise = trans_direction * trans_magnitude
        new_pose[0:3, 3] += trans_noise
        
        # --- 2. 制造随机旋转噪声 ---
        rot_axis = (np.random.rand(3) - 0.5) * 2
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.random.uniform(0, max_rot_deg) * np.pi / 180.0
        noise_rot_matrix = R.from_rotvec(rot_angle * rot_axis).as_matrix()
        new_pose[0:3, 0:3] = noise_rot_matrix @ pose[0:3, 0:3]
        
        # --- 3. 保存新 Pose ---
        save_name = ref_id.replace('.png', '.txt').replace('.jpg', '.txt')
        save_path = os.path.join(output_dir, save_name)
        np.savetxt(save_path, new_pose, fmt='%.6f', delimiter=' ')
        
        # --- 4. 复制对应的 RGB 图像 --- # 👈 新增：图片复制逻辑
        src_rgb_path = os.path.join(rgb_dir, ref_id)
        dst_rgb_path = os.path.join(output_dir, ref_id)
        
        if os.path.exists(src_rgb_path):
            shutil.copy2(src_rgb_path, dst_rgb_path) # copy2 会尽量保留原始文件的元数据
            success_count += 1
        else:
            print(f"  ⚠️ 警告: 找不到对应的图像文件 {src_rgb_path}，已跳过复制该图片。")
        
    print(f"✅ 成功扰动保存了 {len(poses)} 个新 Poses，并成功复制了 {success_count} 张图像！")


def main():
    pairs_file = "/data/nvme0n1/phw/烯创26-0310/outputs/pairs.txt"
    poses_dir = "/data/nvme0n1/phw/烯创26-0310/train/poses"  # 你存放原始 pose 的文件夹
    rgb_dir = "/data/nvme0n1/phw/烯创26-0310/train/rgb"      # 👈 原始图像存放路径
    
    if not os.path.exists(pairs_file):
        print(f"找不到文件: {pairs_file}")
        return

    # 1. 解析 pairs.txt
    ref_ids = []
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ref_ids.append(parts[1])  # 提取第二个 ID (例如 5207.png)
    
    # 2. 读取 Poses
    poses = []
    valid_ref_ids = []
    
    for ref_id in ref_ids:
        pose_filename = ref_id.replace('.png', '.txt').replace('.jpg', '.txt')
        pose_path = os.path.join(poses_dir, pose_filename)
        
        if os.path.exists(pose_path):
            pose_matrix = np.loadtxt(pose_path)
            # 确保是 4x4 矩阵
            if pose_matrix.shape == (4, 4):
                poses.append(pose_matrix)
                valid_ref_ids.append(ref_id)
            else:
                print(f"警告: {pose_filename} 不是 4x4 矩阵！")
        else:
            print(f"警告: 找不到 Pose 文件 {pose_path}")

    poses = np.array(poses)
    
    if len(poses) == 0:
        print("没有成功加载任何 Pose，请检查文件夹路径和文件格式！")
        return

    # 3. 执行诊断分析
    analyze_pose_degeneracy(poses, valid_ref_ids)
    
    # 4. 执行扰动并保存 (传入 rgb_dir)
    # xyz 移动最大 20cm (0.2m), 朝向移动最大 15 度
    perturb_and_save_poses(
        poses=poses, 
        ref_ids=valid_ref_ids, 
        rgb_dir=rgb_dir,           # 👈 把图像源路径传进去
        output_dir="poses_perturbed", 
        max_trans_m=0.20, 
        max_rot_deg=15.0
    )

if __name__ == "__main__":
    main()