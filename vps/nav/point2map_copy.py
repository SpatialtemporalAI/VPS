from optparse import Option
import numpy as np
import open3d as o3d
import os
import math
import time
def segment_and_visualize_point_cloud(pcd_path, camera_height):
    """
    加载点云 -> 根据 camera_height 将点分为高于/低于相机 -> 对低于相机部分用 RANSAC 拟合地面 -> 可视化
    颜色:
      红: 高于相机
      绿: 估计为地板的 inliers
      蓝: 低于相机但非地板的点
    """

    if not os.path.exists(pcd_path):
        print(f"错误：文件未找到在 {pcd_path}")
        return

    # 1. 加载点云
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"读取点云文件失败: {e}")
        return

    points = np.asarray(pcd.points)
    if points.size == 0:
        print("点云为空。")
        return
    num_points = len(points)
    s = time.time()
    # 初始化颜色 (默认灰)
    colors = np.tile(np.array([0.8, 0.8, 0.8], dtype=float), (num_points, 1))

    # --- 2. 高度过滤 ---
    z_coords = points[:, 2]
    idx_above_cam = z_coords > camera_height
    idx_below_cam = ~idx_above_cam  # <= camera_height
    colors[idx_above_cam] = [1.0, 0.0, 0.0]  # 红色：高于相机

    # 选出低于相机的子点云并做索引映射
    below_indices = np.where(idx_below_cam)[0]
    pcd_below_cam = pcd.select_by_index(below_indices.tolist())

    # --- 3. 地板识别（RANSAC） ---
    distance_threshold = 0.02
    ransac_n = 3
    num_iterations = 1000

    if len(pcd_below_cam.points) > ransac_n:
        try:
            plane_model, inliers = pcd_below_cam.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            inliers = np.asarray(inliers, dtype=int)
            # 安全检查 inliers 范围
            inliers = inliers[(inliers >= 0) & (inliers < len(pcd_below_cam.points))]

            if inliers.size == 0:
                # 没有 inliers，全部标为非地板（蓝色）
                print("RANSAC 返回空 inliers，标记所有低于相机的点为非地板（蓝色）")
                colors[idx_below_cam] = [0.0, 0.0, 1.0]
            else:
                # 把 inliers 映射回原始 pcd 索引
                original_indices_floor = below_indices[inliers]
                colors[original_indices_floor] = [0.0, 1.0, 0.0]  # 绿色：地板

                # 高效计算 outliers（相对于 pcd_below_cam）
                mask_below_inliers = np.zeros(len(below_indices), dtype=bool)
                mask_below_inliers[inliers] = True
                outlier_local_idx = np.where(~mask_below_inliers)[0]
                if outlier_local_idx.size > 0:
                    original_indices_non_floor = below_indices[outlier_local_idx]
                    colors[original_indices_non_floor] = [0.0, 0.0, 1.0]  # 蓝色：低于相机但非地板

                print(f"RANSAC 成功：plane_model={plane_model}, floor_inliers={inliers.size}, below_total={len(below_indices)}")

        except Exception as e:
            print(f"RANSAC 拟合失败或点数不足: {e}")
            colors[idx_below_cam] = [0.0, 0.0, 1.0]
    else:
        print("低于相机高度的点数太少，无法进行 RANSAC 地板拟合。全部标记为蓝色。")
        colors[idx_below_cam] = [0.0, 0.0, 1.0]
    e = time.time()
    print(f"RANSAC 时间: {e - s} 秒")
    # 将颜色写回 pcd
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- 5. 改进的可视化（更稳健的视角设置） ---
    print("开始可视化点云...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云分割结果 (红:高于相机, 绿:地板, 蓝:非地板/低于相机)", width=1000, height=700)
    vis.add_geometry(pcd)

    view_control = vis.get_view_control()
    # 使用点云中心作为 lookat
    center = pcd.get_center()
    view_control.set_lookat(center)

    # 设定 front 向量并归一化以避免奇异值
    front = np.array([-1.0, -1.0, -0.5], dtype=float)
    norm = np.linalg.norm(front)
    if norm == 0:
        front = np.array([0.0, 0.0, -1.0])
    else:
        front = front / norm
    view_control.set_front(front.tolist())

    # 明确设定 up 向量（Z向上）
    view_control.set_up([0.0, 0.0, 1.0])

    # 调整 Zoom（0.3 ~ 1.0 取值范围因 open3d 版本略有差异）
    try:
        view_control.set_zoom(0.7)
    except Exception:
        # 某些 open3d 版本对 set_zoom 的参数或存在性敏感，忽略异常
        pass

    # 运行并关闭
    vis.run()
    vis.destroy_window()
pcd_file_path = "/home/phw/visual-localization/VPS/output1.ply" 
camera_z_height = 1.13 

# 调用改进后的可视化函数
segment_and_visualize_point_cloud(pcd_file_path, camera_z_height)