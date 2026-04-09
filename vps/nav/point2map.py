import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Any
import cv2
import yaml
import time
import logging
def get_floor_height(
                pcd: np.ndarray,
                cam_pred_h: float,
                nbins: int = 200,
                peak_frac_cutoff: float = 0.6,
                up: str = 'z'
                ):
    UP_AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}
    
    if up.lower() not in UP_AXIS_MAP:
        raise ValueError(f"Invalid 'up' axis '{up}'. Must be 'x', 'y', or 'z'.")
        
    up_idx = UP_AXIS_MAP[up.lower()]
    height = pcd[:, up_idx]
    height_min, height_max = float(np.min(height)), min(float(np.max(height)),cam_pred_h)
    if height_max - height_min < 1e-9:
        height_peak = height_min
    else:
        bins = np.linspace(height_min, height_max, nbins+1)
        h, _ = np.histogram(height, bins=bins)
        # simple smoothing
        k = max(1, nbins//200)  # small kernel relative to bins
        if k > 1:
            kernel = np.ones(k)/k
            h_s = np.convolve(h.astype(float), kernel, mode='same')
        else:
            h_s = h.astype(float)
        centers = 0.5*(bins[:-1] + bins[1:])
        # choose peaks in the lower portion
        cutoff = height_min + float(peak_frac_cutoff)*(height_max - height_min)
        cand_idxs = np.where(centers <= cutoff)[0]
        if cand_idxs.size == 0:
            peak_idx = int(np.argmax(h_s))
        else:
            local = h_s[cand_idxs]
            pidx = int(np.argmax(local))
            peak_idx = int(cand_idxs[pidx])
        height_peak = float(centers[peak_idx])
    return height_peak

def get_height_scale(cam_pred_h: float,
              height_peak: float,
              cam_real_h: float):
    scale = None
    if cam_real_h is not None:
        pred_cam_to_floor_raw = float(cam_pred_h) - height_peak
        if pred_cam_to_floor_raw <= 0:
            raise ValueError("cam_pred_h must be larger than height_peak_raw (predicted camera above predicted floor)")
        true_cam_to_floor_m = cam_real_h  # in meters
        scale = true_cam_to_floor_m / pred_cam_to_floor_raw  # meters per raw_unit
    return scale


def segment_points_h(  pcd: np.ndarray,
                    floor_height: float,
                    cam_h: float,
                    floor_tol: float = 0.20,
                    up:str = 'z'):
    UP_AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}
    
    if up.lower() not in UP_AXIS_MAP:
        raise ValueError(f"Invalid 'up' axis '{up}'. Must be 'x', 'y', or 'z'.")
        
    up_idx = UP_AXIS_MAP[up.lower()]                
    pts = pcd.copy()
    # obstacles: z > z_floor_m + floor_tol  and z <= cam_h
    obstacle_mask = (pts[:,up_idx] > (floor_height + floor_tol)) & (pts[:,up_idx] <= cam_h)
    available_mask = (pts[:,up_idx] <= (floor_height + floor_tol)) | (pts[:,up_idx] >= floor_height - floor_tol)
    obstacle_pts = pts[obstacle_mask]
    available_pts = pts[available_mask]
    return obstacle_pts,available_pts

# def get_new_occupancy_map(
#     obs_points: np.ndarray, 
#     map_path: str, 
#     yaml_path: str, 
#     occupancy_min_points_per_cell: int = 5,
#     obstacle_value: int = 125,    # 地图上表示障碍物的值 (通常为黑色)
#     free_value: int = 255,      # 地图上表示空闲的值 (通常为白色)
#     smoothing_kernel_size: int = 3 # 用于过滤的内核大小
# ) -> Optional[np.ndarray]:
#     """
#     将新的障碍物点云投影到现有的 Occupancy Map 上进行更新。

#     Args:
#         obs_points (np.ndarray): 形状为 (N, 3) 或 (N, 2) 的障碍物点云数组（已尺度修复）。
#         map_path (str): 现有 Occupancy Map 图像文件的路径 (e.g., .pgm, .png)。
#         yaml_path (str): 对应地图的元数据 YAML 文件路径。
#         obstacle_value (int): 地图上表示“被占据”的像素值 (0-255)。
#         free_value (int): 地图上表示“空闲”的像素值 (0-255)。
#         smoothing_kernel_size (int): 用于最后过滤离散点的内核大小。

#     Returns:
#         Optional[np.ndarray]: 更新后的 Occupancy Map 数组，如果失败则返回 None。
#     """
#     # --- 1. 加载地图和元数据 ---
#     try:
#         # 加载地图图像 (以灰度图读取)
#         occupancy_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
#         if occupancy_map is None:
#             raise FileNotFoundError(f"无法读取地图图像: {map_path}")
        
#         # 加载 YAML 元数据
#         with open(yaml_path, 'r') as f:
#             map_metadata: Dict = yaml.safe_load(f)
            
#         resolution = map_metadata.get('resolution')
#         # 地图左下角的（X, Y）世界坐标
#         origin_world = map_metadata.get('origin') 
        
#         if resolution is None or origin_world is None:
#              raise ValueError("YAML 文件中缺少 'resolution' 或 'origin' 参数。")

#         # 提取地图的原点 X, Y 坐标
#         x_min_world = origin_world[0]
#         y_min_world = origin_world[1]

#     except Exception as e:
#         print(f"加载地图或YAML文件失败: {e}")
#         return None

#     # --- 2. 提取 2D 障碍物点 ---
#     # 我们只关心 X 和 Y 坐标 (前两列)
#     pts_obs_2d = obs_points[:, :2]

#     # --- 3. 世界坐标转栅格索引 ---

#     # 计算世界坐标相对于地图原点的偏移量
#     # grid_x_world = X_world - x_min_world
#     # grid_y_world = Y_world - y_min_world
    
#     # 栅格列 (X轴) 索引: (X_world - x_min) / resolution
#     grid_x = ((pts_obs_2d[:, 0] - x_min_world) / resolution).astype(int)
    
#     # 栅格行 (Y轴) 索引: (Y_world - y_min) / resolution
#     # 注意：ROS/SLAM 地图的 (0, 0) 栅格对应左下角，Y 轴向上。
#     # 数组索引是 (row, col)，row 从上到下递增。
#     # 所以需要将 Y 坐标从左下角原点映射到数组的行索引（从上到下）。
    
#     # 计算地图数组的高度 (行数)
#     map_rows, map_cols = occupancy_map.shape
    
#     # 1. 相对原点的 Y 距离
#     y_dist_from_origin = pts_obs_2d[:, 1] - y_min_world
    
#     # 2. 相对原点的行索引 (Y向上)
#     row_idx_from_origin_bottom = (y_dist_from_origin / resolution).astype(int)
    
#     # 3. 翻转 Y 轴：从底部原点索引转换为数组行索引 (从上到下)
#     # 数组的行索引 = 总行数 - 1 - 底部原点索引
#     grid_y = map_rows - 1 - row_idx_from_origin_bottom 

#     # --- 4. 边界检查和更新地图 ---
    
#     # 边界检查
#     valid_x = (grid_x >= 0) & (grid_x < map_cols)
#     valid_y = (grid_y >= 0) & (grid_y < map_rows)
#     valid_pts = valid_x & valid_y

#     grid_x_valid = grid_x[valid_pts]
#     grid_y_valid = grid_y[valid_pts]
    
#     if valid_pts.size == 0:
#         print("点云中没有点落在地图范围内。")
#         return occupancy_map
    
#     # 4. 计数过滤
#     N = occupancy_min_points_per_cell
#     linear_indices = grid_y_valid * map_cols + grid_x_valid
    
#     # 统计每个栅格的点数
#     count_map = np.bincount(linear_indices, minlength=map_rows * map_cols)
    
#     # 找到点数 >= N 的栅格的线性索引
#     occupied_linear_indices = np.where(count_map >= N)[0]
    
#     # 5. 更新地图
#     if occupied_linear_indices.size > 0:
#         obs_y_indices = occupied_linear_indices // map_cols
#         obs_x_indices = occupied_linear_indices % map_cols
        
#         # 标记为障碍物
#         occupancy_map[obs_y_indices, obs_x_indices] = obstacle_value
        
#         print(f"成功将 {obs_y_indices.size} 个满足阈值 ({N} 个点) 的栅格标记为障碍物。")
        
#     return occupancy_map

#     # 标记被占据的栅格
#     # 你的思路：如果对应位置是空闲（白色），则覆盖成占据（黑色）
#     # 为什么不直接覆盖？因为可能存在置信度更高的现有障碍物。
#     # 这里我们采用“如果点云占据，则标记为占据”的策略。
    
#     # # 获取需要更新的栅格的当前值
#     # current_values = occupancy_map[grid_y_valid, grid_x_valid]
    
#     # # 标记所有点云投影到的栅格为障碍物
#     # occupancy_map[grid_y_valid, grid_x_valid] = obstacle_value
    
#     # print(f"成功将 {len(grid_x_valid)} 个点投影到地图上。")

#     # # --- 5. 过滤离散的黑点 (可选平滑) ---
#     # # 使用形态学闭运算或中值滤波来平滑噪声。
#     # # if smoothing_kernel_size > 1:
#     # #     # 定义一个内核 (例如，用于去除孤立的黑点)
#     # #     kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
        
#     # #     # 中值滤波去除椒盐噪声
#     # #     updated_map = cv2.medianBlur(occupancy_map, smoothing_kernel_size)
        
#     # #     # 如果需要更激进地移除孤立黑点，可以使用开运算 (Open)
#     # #     # updated_map = cv2.morphologyEx(occupancy_map, cv2.MORPH_OPEN, kernel)
        
#     # #     occupancy_map = updated_map

#     # return occupancy_map
def get_new_occupancy_map(
    obs_points: np.ndarray,
    ava_points: np.ndarray, 
    map_path: str, 
    yaml_path: str, 
    camera_6dpose: np.ndarray, # 必须参数
    min_dist: float,          # 新增参数：最小距离
    max_dist: float,          # 新增参数：最大距离
    occupancy_min_points_per_cell: int = 15,
    obstacle_value: int = 50,    # 地图上表示障碍物的值 (通常为黑色)
    free_value: int = 200,      # 地图上表示空闲的值 (通常为白色)
    up: str = 'z',
    showself: bool = True      # 控制是否绘制相机位置并返回彩色图
) -> Optional[np.ndarray]:
    """
    将新的障碍物点云投影到现有的 Occupancy Map 上进行更新，并根据距离过滤点云。
    先是avalibale 表示地板,再是障碍物点云覆盖上去
    如果 showself 为 True,在返回的地图副本上用红点标出相机位置。

    Args:
        obs_points (np.ndarray): 形状为 (N, 3) 或 (N, 2) 的障碍物点云数组（已尺度修复）。
        ava_points (np.ndarray): 形状为 (N, 3) 或 (N, 2) 的地板区域云数组（已尺度修复）。
        map_path (str): 现有 Occupancy Map 图像文件的路径 (e.g., .pgm, .png)。
        yaml_path (str): 对应地图的元数据 YAML 文件路径。
        camera_6dpose (np.ndarray): 4x4 齐次矩阵，用于确定自身位置和过滤距离。
        min_dist (float): 点云到相机在地图平面的最小欧氏距离。
        max_dist (float): 点云到相机在地图平面的最大欧氏距离。
        occupancy_min_points_per_cell (int): 单格点数阈值，>= 则视为占据。
        obstacle_value (int): 地图上表示“被占据”的像素值 (0-255)。
        free_value (int): 地图上表示“空闲”的像素值 (0-255)。
        up (str): 确定地图平面的轴 ('z' 或 'y')。
        showself (bool): 是否在返回的地图上用红点标出相机位置。

    Returns:
        Optional[np.ndarray]: 更新后的 Occupancy Map 数组（如果 showself 为 False 则为灰度图 uint8,
                              否则返回 BGR 彩色图 uint8)。失败则返回 None。
    """
    # 映射关系
    start_time = time.time()
    UP_AXIS_TO_PLANE = {
    'z': (0, 1), # P1=X(0), P2=Y(1)
    'y': (0, 2), # P1=X(0), P2=Z(2)
    }
    up_lower = up.lower()
    if up_lower not in UP_AXIS_TO_PLANE:
        print(f"错误: 不支持的 'up' 轴 '{up}'. 必须是 'z' 或 'y'.")
        return None
        
    p1_idx, p2_idx = UP_AXIS_TO_PLANE[up_lower]
    
    # --- 1. 加载地图和元数据 ---
    # ... (加载地图和 YAML 元数据的代码保持不变) ...
    try:
        occupancy_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if occupancy_map is None:
            raise FileNotFoundError(f"无法读取地图图像: {map_path}")
        
        with open(yaml_path, 'r') as f:
            map_metadata: Dict = yaml.safe_load(f)
            
        resolution = map_metadata.get('resolution')
        origin_world = map_metadata.get('origin') 
        
        if resolution is None or origin_world is None:
             raise ValueError("YAML 文件中缺少 'resolution' 或 'origin' 参数。")

        x_min_world = origin_world[0]
        y_min_world = origin_world[1]

    except Exception as e:
        print(f"加载地图或YAML文件失败: {e}")
        return None
    logging.info(f"到加载地图和元数据的时间：{time.time() - start_time:.4f}s")
    # ----------------------------------------

    # --- 2. 可行区域距离过滤 ---
    
    # 获取相机在地图平面上的世界坐标 (P1, P2 轴)
    cam_p1_world = camera_6dpose[p1_idx, 3]
    cam_p2_world = camera_6dpose[p2_idx, 3]

    # 提取点云的 P1 和 P2 坐标
    pts_p1_world = ava_points[:, p1_idx]
    pts_p2_world = ava_points[:, p2_idx]
    
    # 计算点云到相机的欧氏距离 (在 P1-P2 平面上)
    dist_sq = (pts_p1_world - cam_p1_world)**2 + (pts_p2_world - cam_p2_world)**2
    distances = np.sqrt(dist_sq)

    # 过滤距离在 [min_dist, max_dist] 之间的点
    valid_dist_mask = (distances >= min_dist) & (distances <= max_dist)
    
    # 提取过滤后的点云
    ava_points_filtered = ava_points[valid_dist_mask]
    
    logging.info(f"到可行区域距离过滤的时间：{time.time() - start_time:.4f}s")
    # --- 3. 提取 2D 可行区域 ---
    ava_points_2d = ava_points_filtered[:, [p1_idx,p2_idx]]
    
    # --- 4. 世界坐标转栅格索引 ---
    
    # 栅格列 (P1轴) 索引
    grid_x = ((ava_points_2d[:, 0] - x_min_world) / resolution).astype(int)
    
    map_rows, map_cols = occupancy_map.shape
    
    # 栅格行 (P2轴) 索引
    y_dist_from_origin = ava_points_2d[:, 1] - y_min_world
    row_idx_from_origin_bottom = (y_dist_from_origin / resolution).astype(int)
    grid_y = map_rows - 1 - row_idx_from_origin_bottom 

    # --- 5. 边界检查和更新地图 ---
    
    # 边界检查
    valid_x = (grid_x >= 0) & (grid_x < map_cols)
    valid_y = (grid_y >= 0) & (grid_y < map_rows)
    valid_pts = valid_x & valid_y

    grid_x_valid = grid_x[valid_pts]
    grid_y_valid = grid_y[valid_pts]
    
    
    # 计数过滤和更新
    N = occupancy_min_points_per_cell
    
    if valid_pts.size > 0:
        linear_indices = grid_y_valid * map_cols + grid_x_valid
        count_map = np.bincount(linear_indices, minlength=map_rows * map_cols)
        occupied_linear_indices = np.where(count_map >= N)[0]
        
        if occupied_linear_indices.size > 0:
            ava_y_indices = occupied_linear_indices // map_cols
            ava_x_indices = occupied_linear_indices % map_cols
            
            occupancy_map[ava_y_indices, ava_x_indices] = free_value
            
    logging.info(f"到可行区域更新地图的时间：{time.time() - start_time:.4f}s")
    # --- 2. 障碍物
    # 距离过滤 ---
    
    # 获取相机在地图平面上的世界坐标 (P1, P2 轴)
    cam_p1_world = camera_6dpose[p1_idx, 3]
    cam_p2_world = camera_6dpose[p2_idx, 3]

    # 提取点云的 P1 和 P2 坐标
    pts_p1_world = obs_points[:, p1_idx]
    pts_p2_world = obs_points[:, p2_idx]
    
    # 计算点云到相机的欧氏距离 (在 P1-P2 平面上)
    dist_sq = (pts_p1_world - cam_p1_world)**2 + (pts_p2_world - cam_p2_world)**2
    distances = np.sqrt(dist_sq)

    # 过滤距离在 [min_dist, max_dist] 之间的点
    valid_dist_mask = (distances >= min_dist) & (distances <= max_dist)
    
    # 提取过滤后的点云
    obs_points_filtered = obs_points[valid_dist_mask]
    
    if obs_points_filtered.size == 0:
        print(f"点云经过距离过滤 (min={min_dist}, max={max_dist}) 后为空。")
        # 即使点云为空，仍需继续到绘制相机流程
    
    # --- 3. 提取 2D 障碍物点 ---
    pts_obs_2d = obs_points_filtered[:, [p1_idx,p2_idx]]
    
    # --- 4. 世界坐标转栅格索引 ---
    
    # 栅格列 (P1轴) 索引
    grid_x = ((pts_obs_2d[:, 0] - x_min_world) / resolution).astype(int)
    
    map_rows, map_cols = occupancy_map.shape
    
    # 栅格行 (P2轴) 索引
    y_dist_from_origin = pts_obs_2d[:, 1] - y_min_world
    row_idx_from_origin_bottom = (y_dist_from_origin / resolution).astype(int)
    grid_y = map_rows - 1 - row_idx_from_origin_bottom 

    # --- 5. 边界检查和更新地图 ---
    
    # 边界检查
    valid_x = (grid_x >= 0) & (grid_x < map_cols)
    valid_y = (grid_y >= 0) & (grid_y < map_rows)
    valid_pts = valid_x & valid_y

    grid_x_valid = grid_x[valid_pts]
    grid_y_valid = grid_y[valid_pts]
    
    if valid_pts.size == 0:
        print("经过距离和边界过滤后，没有点落在地图范围内。")
    logging.info(f"到障碍物距离过滤的时间：{time.time() - start_time:.4f}s")

    # 计数过滤和更新
    N = occupancy_min_points_per_cell
    
    if valid_pts.size > 0:
        linear_indices = grid_y_valid * map_cols + grid_x_valid
        count_map = np.bincount(linear_indices, minlength=map_rows * map_cols)
        occupied_linear_indices = np.where(count_map >= N)[0]
        
        if occupied_linear_indices.size > 0:
            obs_y_indices = occupied_linear_indices // map_cols
            obs_x_indices = occupied_linear_indices % map_cols
            
            occupancy_map[obs_y_indices, obs_x_indices] = obstacle_value
            
            print(f"成功将 {obs_y_indices.size} 个满足阈值 ({N} 个点) 的栅格标记为障碍物。")
    logging.info(f"到障碍物更新地图的时间：{time.time() - start_time:.4f}s")

    # # ====== 在这里做 10cm 障碍物膨胀 ======
    # inflation_pixels = int(0.30 / resolution)  # 推荐写成这样，更鲁棒
    # inflation_pixels = max(1, inflation_pixels)

    # obstacle_mask = (occupancy_map == obstacle_value).astype(np.uint8)

    # kernel = cv2.getStructuringElement(
    #     cv2.MORPH_ELLIPSE,
    #     (2 * inflation_pixels + 1, 2 * inflation_pixels + 1)
    # )

    # inflated_mask = cv2.dilate(obstacle_mask, kernel)
    # occupancy_map[inflated_mask > 0] = obstacle_value

    # logging.info(
    #     f"完成障碍物膨胀：{inflation_pixels} px "
    #     f"(≈ {inflation_pixels * resolution:.2f} m)"
    # )
    # =====================================
    # --- 6. （可选）绘制相机位置 ---
    # 只有当 showself 为 True 时，才绘制红点并返回彩色图
    if not showself:
        return occupancy_map

    # 世界坐标 -> 栅格坐标（相机位置）
    grid_x_cam = int((cam_p1_world - x_min_world) / resolution)
    row_idx_from_origin_bottom_cam = int((cam_p2_world - y_min_world) / resolution)
    grid_y_cam = map_rows - 1 - row_idx_from_origin_bottom_cam

    # 检查相机是否越界
    if not (0 <= grid_x_cam < map_cols and 0 <= grid_y_cam < map_rows):
        print(f"相机位置 ({cam_p1_world:.3f}, {cam_p2_world:.3f}) 超出地图范围，跳过绘制。")
        return occupancy_map

    # 将灰度图转换为 BGR 彩色图（副本）
    colored_map = cv2.cvtColor(occupancy_map.copy(), cv2.COLOR_GRAY2BGR)

    # 绘制红点
    radius = max(2, int(min(map_rows, map_cols) / 200))
    cv2.circle(colored_map, (grid_x_cam, grid_y_cam), radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    print(f"已在返回的彩色地图上用红点标注相机位置，像素位置: ({grid_x_cam}, {grid_y_cam})，半径: {radius}")

    return colored_map

# --------------------
# Usage example
# --------------------
if __name__ == "__main__":
    pcd_path = "/home/phw/visual-localization/VPS/output1.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.asarray(pcd.points)
    cam_pred_h = 1.13
    cam_real_h = 1.5
    pred_floor_h = get_floor_height(pcd,cam_pred_h)
    print(pred_floor_h)
    scale = get_height_scale(cam_pred_h,pred_floor_h,cam_real_h)
    obs = get_obstacles_points(pcd,pred_floor_h,cam_pred_h)
    map_path = "/home/phw/visual-localization/VPS/nav/map.png"
    yaml_path = "/home/phw/visual-localization/VPS/nav/map.yaml"
    new_map = get_new_occupancy_map(obs,map_path,yaml_path)
    cv2.imshow("dawd",new_map)
    cv2.waitKey(0)

# 当用户按下任意键后，程序执行到这里，并关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
    

