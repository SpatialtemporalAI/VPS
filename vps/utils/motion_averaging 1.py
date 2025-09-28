import numpy as np
import torch
import os
from typing import Optional
from scipy.spatial.transform import Rotation as R
from .lud_localization import LUDLocalizer, convert_lines_to_lud_format
# from pdb import set_trace as bb


class MotionAveraging:
    def __init__(self, use_lud=True, lud_loss_type="huber"):
        super(MotionAveraging, self).__init__()
        self.use_lud = use_lud
        self.lud_loss_type = lud_loss_type
        
        # 初始化LUD定位器
        if self.use_lud:
            self.lud_localizer = LUDLocalizer(
                loss_type=lud_loss_type,
                huber_delta=1.0,
                max_iterations=1000,
                tolerance=1e-6,
                verbose=False
            )

    def rotation_averaging(self, matrices): 
        """
        Args:
            matrices (np.ndarray): shape of (N, 3, 3), absolute rotation matrices 
        Returns:
            avg_rotation_matrix (np.ndarray): shape of (3, 3), absolute rotation matrix
        """
        quaternions = [R.from_matrix(mat).as_quat() for mat in matrices]
        quaternions = np.array(quaternions)
        # ret_quaternion = np.mean(quaternions, axis=0)
        ret_quaternion = np.median(quaternions, axis=0)  # slightly better than mean
        norm = np.linalg.norm(ret_quaternion)
        ret_quaternion /= norm
        avg_rotation_matrix = R.from_quat(ret_quaternion).as_matrix()
        return avg_rotation_matrix


    def rotation_averaging_enhanced(self, matrices):
        """
        两阶段旋转平均化：先筛除离群值，再基于一致性加权优化
        
        Args:
            matrices (np.ndarray): shape of (N, 3, 3), absolute rotation matrices 
        Returns:
            avg_rotation_matrix (np.ndarray): shape of (3, 3), absolute rotation matrix
        """
        quaternions = [R.from_matrix(mat).as_quat() for mat in matrices]
        quaternions = np.array(quaternions)
        
        # 第一阶段：筛除离群值
        filtered_quats = self._remove_outliers_stage1(quaternions)
        
        # 第二阶段：基于一致性加权优化
        final_result = self._weighted_averaging_stage2(quaternions, filtered_quats)
        
        return final_result

    def _remove_outliers_stage1(self, quaternions):
        """第一阶段：移除角度偏差最大的离群值"""
        if len(quaternions) <= 3:
            return quaternions  # 太少的话直接返回
        
        # 计算每个四元数到其他所有四元数的平均角距离
        distances = []
        for i, quat_i in enumerate(quaternions):
            total_angle = 0
            count = 0
            for j, quat_j in enumerate(quaternions):
                if i != j:
                    # 计算四元数之间的角距离
                    dot_product = np.abs(np.dot(quat_i, quat_j))
                    angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
                    total_angle += angle_dist
                    count += 1
            avg_angle = total_angle / count
            distances.append(avg_angle)
        
        # 移除平均角度最大的1-2个（根据数据量决定）
        remove_count = min(2, len(quaternions) // 3)  # 最多移除1/3
        if remove_count > 0:
            # 找到距离最大的几个索引
            worst_indices = np.argsort(distances)[-remove_count:]
            # 保留其他索引
            good_indices = [i for i in range(len(quaternions)) if i not in worst_indices]
            return quaternions[good_indices]
        
        return quaternions

    def _weighted_averaging_stage2(self, all_quats, filtered_quats):
        """第二阶段：基于一致性计算权重并加权平均"""
        # 使用第一阶段的结果作为参考
        if len(filtered_quats) > 0:
            reference_quat = np.mean(filtered_quats, axis=0)
            # reference_quat = np.median(filtered_quats, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        else:
            # 如果没有过滤结果，使用所有四元数的中位数
            reference_quat = np.median(all_quats, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        
        # 计算每个四元数到参考四元数的权重
        weights = []
        for quat in all_quats:
            # 确保四元数方向一致
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            
            # 计算角距离
            dot_product = np.abs(np.dot(reference_quat, quat))
            angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
            
            # 权重：角度越小，权重越大（使用高斯权重）
            weight = np.exp(-angle_dist / 0.3)  # 0.3是控制参数，可调整
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        # 加权平均
        weighted_quat = np.zeros(4)
        for quat, weight in zip(all_quats, weights):
            # 再次确保方向一致
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            weighted_quat += weight * quat
        
        # 归一化最终结果
        weighted_quat /= np.linalg.norm(weighted_quat)
        return R.from_quat(weighted_quat).as_matrix()

    def camera_center_triangulation(self, points):  
        """
        Args:
            points (np.ndarray): shape of (N, 2, 3), 3D coordinates of the 2 endpoints of N lines (directions)
        Returns:
            x (np.ndarray): shape of (3), least squares intersection point
        """
        n = points.shape[0]
        p = points[:, 0, :]
        q = points[:, 1, :]
        d = q - p
        d_norm_sq = np.sum(d ** 2, axis=1, keepdims=True)
        eye_3 = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(n, axis=0)
        d_dT = np.expand_dims(d, axis=2) * np.expand_dims(d, axis=1)
        A_blocks = eye_3 - d_dT / d_norm_sq.reshape(n, 1, 1)
        A = A_blocks.reshape(-1, 3)
        b_blocks = np.matmul(eye_3 - d_dT / d_norm_sq.reshape(n, 1, 1), np.expand_dims(p, axis=2))
        b = b_blocks.reshape(-1)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / S)
        x = np.dot(Vt.T, np.dot(S_inv, np.dot(U.T, b)))
        return x

    def camera_center_triangulation_torch(self, points):  
        """
        Args:
            points (torch.Tensor): shape of (N, 2, 3), 3D coordinates of the 2 endpoints of N lines (directions)
        Returns:
            x (torch.Tensor): shape of (3), least squares intersection point
        """
        n = points.shape[0]
        p = points[:, 0, :]
        q = points[:, 1, :]
        d = q - p
        d_norm_sq = (d ** 2).sum(dim=1, keepdim=True)
        eye_3 = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        d_dT = d.unsqueeze(2) * d.unsqueeze(1)
        A_blocks = eye_3 - d_dT / d_norm_sq.unsqueeze(2)
        A = A_blocks.reshape(-1, 3)
        b_blocks = (eye_3 - d_dT / d_norm_sq.unsqueeze(2)) @ p.unsqueeze(2)
        b = b_blocks.reshape(-1)
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        S_inv = torch.diag(1 / S)
        x = Vt.T @ (S_inv @ (U.T @ b))
        return x

    def lud_camera_center_estimation(self, lines: np.ndarray, weights: Optional[np.ndarray] = None,
                                   initial_guess: Optional[np.ndarray] = None, debug: bool = False) -> np.ndarray:
        """
        使用LUD算法进行鲁棒的相机中心估计
        
        Args:
            lines: (N, 2, 3) 射线数据，每条射线由两个3D点定义
            weights: (N,) 每条射线的权重
            initial_guess: (3,) 初始位置猜测
            debug: 是否输出调试信息
            
        Returns:
            estimated_center: (3,) 估计的相机中心
        """
        if not self.use_lud:
            # 如果未启用LUD，回退到传统方法
            return self.camera_center_triangulation(lines)
        
        try:
            # 转换数据格式
            directions, reference_positions = convert_lines_to_lud_format(lines)
            
            # 使用LUD算法估计位置
            if debug:
                self.lud_localizer.verbose = True
                
            result = self.lud_localizer.estimate_location(
                directions=directions,
                reference_positions=reference_positions,
                weights=weights,
                initial_guess=initial_guess
            )
            
            if debug:
                print(f"LUD估计完成: method={result['method']}, cost={result['cost']:.6f}")
                print(f"平均残差: {np.mean(result['residuals']):.6f}")
                self.lud_localizer.verbose = False
                
            return result['position']
            
        except Exception as e:
            if debug:
                print(f"LUD估计失败，回退到传统方法: {e}")
            # 如果LUD失败，回退到传统三角化
            return self.camera_center_triangulation(lines)

    # def motion_averaging(self, poses_db, poses_q2d): 
    #     """
    #     Args:
    #         poses_db (List): list of (4, 4) absolute poses that transform points from camera to world
    #         poses_q2d (List): list of (4, 4) relative poses that transform query to dbs
    #     Returns:
    #         Rt (np.ndarray): shape of (4, 4), absolute pose of query 
    #     """
    #     assert len(poses_db) == len(poses_q2d)
    #     qR = []
    #     lines = []
    #     for pid in range(len(poses_db)):
    #         pose_q = poses_db[pid] @ poses_q2d[pid]
    #         qR.append(pose_q[0:3,0:3])
    #         p_beg = poses_db[pid][0:3,3]
    #         p_end = (poses_db[pid] @ poses_q2d[pid][0:4,3])[0:3]
    #         endpoints = np.concatenate((p_beg[None,...], p_end[None,...]), axis=0)
    #         lines.append(endpoints)
        
    #     # rotation averaging
    #     # avg_qR = self.rotation_averaging(np.array(qR))
    #     avg_qR = self.rotation_averaging_enhanced(np.array(qR))
    #     # camera center triangulation
    #     cam_cen = self.camera_center_triangulation(np.array(lines))
    #     # cam_cen = self.camera_center_triangulation_torch(torch.tensor(np.array(lines)))
        
    #     Rt = np.identity(4)
    #     Rt[0:3,0:3] = avg_qR
    #     Rt[0:3,3] = cam_cen
    #     return Rt



    def motion_averaging(self, poses_db, poses_q2d, debug=False): 
        """
        针对小样本数据（如10对）优化的运动平均算法
        Args:
            poses_db (List): list of (4, 4) absolute poses that transform points from camera to world
            poses_q2d (List): list of (4, 4) relative poses that transform query to dbs
            debug (bool): 是否输出调试信息
        Returns:
            Rt (np.ndarray): shape of (4, 4), absolute pose of query 
        """
        assert len(poses_db) == len(poses_q2d)
        n_pairs = len(poses_db)
        
        if debug:
            print(f"运动平均：处理 {n_pairs} 对数据")
        
        qR = []
        lines = []
        pose_qualities = []  # 记录每个pose的质量指标
        
        for pid in range(len(poses_db)):
            pose_q = poses_db[pid] @ poses_q2d[pid]
            qR.append(pose_q[0:3,0:3])
            p_beg = poses_db[pid][0:3,3]
            p_end = (poses_db[pid] @ poses_q2d[pid][0:4,3])[0:3]
            endpoints = np.concatenate((p_beg[None,...], p_end[None,...]), axis=0)
            lines.append(endpoints)
            
            # 计算pose质量指标（基于基线长度和旋转幅度）
            try:
                baseline = np.linalg.norm(p_end - p_beg)
                # 确保旋转矩阵有效
                R_q2d = poses_q2d[pid][:3,:3]
                trace_val = np.trace(R_q2d)
                trace_val = np.clip(trace_val, -1, 3)  # 理论上应该在[-1, 3]范围内
                rot_angle = np.arccos(np.clip((trace_val - 1) / 2, -1, 1))
                
                # 使用更鲁棒的质量计算
                quality = baseline * (np.sin(rot_angle + 0.1) + 0.1)  # 避免除零和负值
                quality = np.clip(quality, 0.01, 10.0)  # 限制在合理范围内
                
                # 检查是否为有效数值
                if not np.isfinite(quality):
                    quality = 0.1  # 默认质量值
                    
            except Exception as e:
                if debug:
                    print(f"Pose {pid} 质量计算异常: {e}, 使用默认值")
                quality = 0.1  # 默认质量值
                
            pose_qualities.append(quality)
        
        pose_qualities = np.array(pose_qualities)
        
        # 小样本优化策略
        if n_pairs <= 12:
            # 对于小样本，使用更保守的过滤和更精细的权重计算
            qR_array = np.array(qR)
            filtered_indices, weights = self._get_small_sample_filtering_and_weights(
                qR_array, pose_qualities, debug=debug
            )
            
            # 使用多种方法融合的旋转平均
            avg_qR = self._multi_method_rotation_averaging(
                qR_array, filtered_indices, weights, debug=debug
            )
            
            # 增强的相机中心三角化
            cam_cen = self._enhanced_camera_center_triangulation(
                np.array(lines), filtered_indices, weights, pose_qualities, debug=debug
            )
        else:
            # 大样本使用原有方法
            qR_array = np.array(qR)
            filtered_indices, weights = self._get_rotation_filtering_and_weights(qR_array)
            filtered_qR = qR_array[filtered_indices]
            avg_qR = self._weighted_rotation_averaging(filtered_qR, weights[filtered_indices])
            cam_cen = self._robust_weighted_camera_center_triangulation(
                np.array(lines), filtered_indices, weights
            )
        
        if debug:
            print(f"使用了 {len(filtered_indices)}/{n_pairs} 个pose")
            print(f"平均权重: {np.mean(weights[filtered_indices]):.4f}")
        
        Rt = np.identity(4)
        Rt[0:3,0:3] = avg_qR
        Rt[0:3,3] = cam_cen
        return Rt

    def _get_rotation_filtering_and_weights(self, qR_array):
        """获取旋转过滤结果和权重，用于后续的位移优化"""
        quaternions = [R.from_matrix(mat).as_quat() for mat in qR_array]
        quaternions = np.array(quaternions)
        
        # 第一阶段：筛除离群值
        filtered_indices = self._get_filtered_indices(quaternions)
        
        # 第二阶段：计算权重
        weights = self._compute_rotation_weights(quaternions, filtered_indices)
        
        return filtered_indices, weights

    def _get_filtered_indices(self, quaternions):
        """获取过滤后的索引"""
        if len(quaternions) <= 3:
            return list(range(len(quaternions)))
        
        # 计算每个四元数的平均角距离
        distances = []
        for i, quat_i in enumerate(quaternions):
            total_angle = 0
            count = 0
            for j, quat_j in enumerate(quaternions):
                if i != j:
                    dot_product = np.abs(np.dot(quat_i, quat_j))
                    angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
                    total_angle += angle_dist
                    count += 1
            avg_angle = total_angle / count
            distances.append(avg_angle)
        
        # 移除平均角度最大的1-2个
        remove_count = min(2, len(quaternions) // 3)
        if remove_count > 0:
            worst_indices = np.argsort(distances)[-remove_count:]
            good_indices = [i for i in range(len(quaternions)) if i not in worst_indices]
            return good_indices
        
        return list(range(len(quaternions)))

    def _compute_rotation_weights(self, quaternions, filtered_indices):
        """计算基于旋转一致性的权重"""
        if len(filtered_indices) > 0:
            # 使用过滤后的四元数计算参考值
            filtered_quats = quaternions[filtered_indices]
            reference_quat = np.mean(filtered_quats, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        else:
            reference_quat = np.median(quaternions, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        
        # 计算每个四元数的权重
        weights = []
        for quat in quaternions:
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            
            dot_product = np.abs(np.dot(reference_quat, quat))
            angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
            
            # 高斯权重：角度越小，权重越大
            weight = np.exp(-angle_dist / 0.3)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        return weights

    def _weighted_rotation_averaging(self, filtered_qR, weights):
        """对过滤后的旋转矩阵进行加权平均"""
        quaternions = [R.from_matrix(mat).as_quat() for mat in filtered_qR]
        quaternions = np.array(quaternions)
        
        # 归一化权重
        weights = weights / np.sum(weights)
        
        # 加权平均
        weighted_quat = np.zeros(4)
        reference_quat = np.median(quaternions, axis=0)
        reference_quat /= np.linalg.norm(reference_quat)
        
        for quat, weight in zip(quaternions, weights):
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            weighted_quat += weight * quat
        
        # 安全的归一化
        quat_norm = np.linalg.norm(weighted_quat)
        if quat_norm < 1e-8:
            weighted_quat = np.array([0, 0, 0, 1])  # 单位四元数
        else:
            weighted_quat /= quat_norm
            
        return R.from_quat(weighted_quat).as_matrix()

    def _weighted_camera_center_triangulation(self, lines, filtered_indices, weights):
        """基于旋转一致性的加权相机中心三角化"""
        if len(filtered_indices) == 0:
            return self.camera_center_triangulation(lines)
        
        # 使用过滤后的直线进行初步估计
        filtered_lines = lines[filtered_indices]
        initial_center = self.camera_center_triangulation(filtered_lines)
        
        # 基于权重进行精细优化
        n = lines.shape[0]
        p = lines[:, 0, :]
        q = lines[:, 1, :]
        d = q - p
        
        # 构建加权最小二乘问题
        weighted_A = []
        weighted_b = []
        
        for i in range(n):
            weight = np.sqrt(weights[i])  # 权重平方根用于加权最小二乘
            
            d_i = d[i]
            p_i = p[i]
            d_norm_sq = np.sum(d_i ** 2)
            
            # 投影矩阵
            P = np.eye(3) - np.outer(d_i, d_i) / d_norm_sq
            
            # 加权矩阵块
            weighted_A.append(weight * P)
            weighted_b.append(weight * (P @ p_i))
        
        # 组合加权系统
        A = np.vstack(weighted_A)
        b = np.concatenate(weighted_b)
        
        # 求解加权最小二乘
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / S)
        x = Vt.T @ (S_inv @ (U.T @ b))
        
        return x

    def _robust_weighted_camera_center_triangulation(self, lines, filtered_indices, rot_weights,
                                                     max_iters=10, tol=1e-6, huber_delta=None,
                                                     angle_power=1.0):
        """
        IRLS鲁棒加权三角化:综合旋转权重、线间夹角权重、残差权重
        Args:
            lines (np.ndarray): (N, 2, 3)
            filtered_indices (List[int]): 旋转筛除后的内点索引
            rot_weights (np.ndarray): (N,) 来自旋转一致性的权重
        Returns:
            x (np.ndarray): (3,)
        """
        n = lines.shape[0]
        p = lines[:, 0, :].astype(np.float64)
        q = lines[:, 1, :].astype(np.float64)
        d = q - p
        # 归一化方向，避免尺度影响
        d_norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        d_unit = d / d_norm

        # 投影矩阵 Pi = I - d d^T
        I3 = np.eye(3, dtype=np.float64)
        P_list = I3[None, :, :] - np.einsum('ni,nj->nij', d_unit, d_unit)
        Pp = np.einsum('nij,ni->nj', P_list, p)  # Pi @ p_i

        # 线间夹角一致性权重：与其他线条越正交，权重越大
        # u_i = mean_j sin(theta_ij)
        dot = np.clip(d_unit @ d_unit.T, -1.0, 1.0)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - dot ** 2))
        np.fill_diagonal(sin_theta, 0.0)
        angle_uniqueness = np.mean(sin_theta, axis=1)
        angle_weights = np.power(angle_uniqueness + 1e-6, angle_power)
        angle_weights /= (np.max(angle_weights) + 1e-12)

        # 初始解：使用过滤后的子集与组合权重解一次加权最小二乘
        base_weights = rot_weights.astype(np.float64) * angle_weights
        base_weights /= (np.sum(base_weights) + 1e-12)

        if len(filtered_indices) > 0:
            idx = np.array(filtered_indices, dtype=int)
        else:
            idx = np.arange(n, dtype=int)

        sqrt_w0 = np.sqrt(base_weights[idx])[:, None, None]
        A0 = (sqrt_w0 * P_list[idx]).reshape(-1, 3)
        b0 = (sqrt_w0[:, :, 0] * Pp[idx]).reshape(-1)
        U, S, Vt = np.linalg.svd(A0, full_matrices=False)
        S_inv = np.diag(1.0 / (S + 1e-12))
        x = Vt.T @ (S_inv @ (U.T @ b0))

        # IRLS
        for _ in range(max_iters):
            # 残差：到直线的垂直距离向量
            r_vec = (P_list @ x[:, None]).squeeze(-1) - Pp
            r = np.linalg.norm(r_vec, axis=1)

            # 动态Huber阈值（基于MAD）
            if huber_delta is None:
                med = np.median(r)
                mad = np.median(np.abs(r - med)) + 1e-12
                delta = 1.4826 * mad + 1e-12
            else:
                delta = float(huber_delta)

            # Huber权重
            w_res = np.ones_like(r)
            large = r > delta
            w_res[large] = delta / (r[large] + 1e-12)

            # 组合权重
            w = base_weights * w_res
            w /= (np.sum(w) + 1e-12)

            # 解新的加权最小二乘
            sqrt_w = np.sqrt(w)[:, None, None]
            A = (sqrt_w * P_list).reshape(-1, 3)
            b = (sqrt_w[:, :, 0] * Pp).reshape(-1)
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            S_inv = np.diag(1.0 / (S + 1e-12))
            x_new = Vt.T @ (S_inv @ (U.T @ b))

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        return x

    def _get_small_sample_filtering_and_weights(self, qR_array, pose_qualities, debug=False):
        """针对小样本的保守过滤和精细权重计算"""
        n = len(qR_array)
        quaternions = [R.from_matrix(mat).as_quat() for mat in qR_array]
        quaternions = np.array(quaternions)
        
        if debug:
            print(f"Pose质量分布: min={pose_qualities.min():.4f}, max={pose_qualities.max():.4f}")
        
        # 小样本策略：更保守的过滤
        if n <= 6:
            # 极小样本，只移除最明显的离群值
            filtered_indices = self._conservative_outlier_removal(quaternions, max_remove=1)
        elif n <= 10:
            # 小样本，移除最多2个离群值
            filtered_indices = self._conservative_outlier_removal(quaternions, max_remove=2)
        else:
            # 中等样本，移除最多3个离群值
            filtered_indices = self._conservative_outlier_removal(quaternions, max_remove=3)
        
        # 综合权重：旋转一致性 + pose质量 + 几何一致性
        weights = self._compute_comprehensive_weights(
            quaternions, pose_qualities, filtered_indices, debug=debug
        )
        
        return filtered_indices, weights

    def _conservative_outlier_removal(self, quaternions, max_remove=2):
        """保守的离群值移除策略"""
        n = len(quaternions)
        if n <= 3:
            return list(range(n))
        
        # 计算每个四元数到其他所有四元数的中位数角距离
        median_distances = []
        for i, quat_i in enumerate(quaternions):
            distances = []
            for j, quat_j in enumerate(quaternions):
                if i != j:
                    dot_product = np.abs(np.dot(quat_i, quat_j))
                    angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
                    distances.append(angle_dist)
            median_dist = np.median(distances)
            median_distances.append(median_dist)
        
        # 只移除明显的离群值（距离超过阈值）
        threshold = np.percentile(median_distances, 75) + 1.5 * (
            np.percentile(median_distances, 75) - np.percentile(median_distances, 25)
        )
        
        outlier_candidates = [i for i, dist in enumerate(median_distances) if dist > threshold]
        
        # 限制移除数量
        if len(outlier_candidates) > max_remove:
            # 选择距离最大的几个
            outlier_indices = sorted(outlier_candidates, key=lambda x: median_distances[x])[-max_remove:]
        else:
            outlier_indices = outlier_candidates
        
        good_indices = [i for i in range(n) if i not in outlier_indices]
        return good_indices

    def _compute_comprehensive_weights(self, quaternions, pose_qualities, filtered_indices, debug=False):
        """计算综合权重：旋转一致性 + pose质量 + 几何稳定性"""
        n = len(quaternions)
        
        # 1. 旋转一致性权重
        if len(filtered_indices) > 0:
            filtered_quats = quaternions[filtered_indices]
            reference_quat = np.median(filtered_quats, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        else:
            reference_quat = np.median(quaternions, axis=0)
            reference_quat /= np.linalg.norm(reference_quat)
        
        rotation_weights = []
        for quat in quaternions:
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            dot_product = np.abs(np.dot(reference_quat, quat))
            angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
            # 使用更尖锐的权重函数
            angle_threshold = getattr(self, 'angle_threshold', 0.2)
            weight = np.exp(-angle_dist / angle_threshold)
            rotation_weights.append(weight)
        rotation_weights = np.array(rotation_weights)
        
        # 2. Pose质量权重（归一化）
        quality_weights = pose_qualities / (np.max(pose_qualities) + 1e-8)
        quality_weights = np.clip(quality_weights, 0.01, 1.0)  # 确保在合理范围内
        quality_weights = np.power(quality_weights, 0.5)  # 平方根，减少极端值影响
        
        # 3. 几何稳定性权重（基于四元数的分布密度）
        stability_weights = []
        for i, quat_i in enumerate(quaternions):
            # 计算到最近邻居的距离
            min_dist = float('inf')
            for j, quat_j in enumerate(quaternions):
                if i != j:
                    dot_product = np.abs(np.dot(quat_i, quat_j))
                    angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
                    min_dist = min(min_dist, angle_dist)
            # 距离越小（越聚集），权重越高
            stability_weight = np.exp(-min_dist / 0.3)
            stability_weights.append(stability_weight)
        stability_weights = np.array(stability_weights)
        
        # 综合权重（可调整各项的重要性）
        alpha = getattr(self, 'rotation_weight', 0.5)
        beta = getattr(self, 'quality_weight', 0.3) 
        gamma = getattr(self, 'stability_weight', 0.2)
        combined_weights = (alpha * rotation_weights + 
                          beta * quality_weights + 
                          gamma * stability_weights)
        
        # 归一化
        combined_weights /= (np.sum(combined_weights) + 1e-8)
        
        if debug:
            print(f"权重分布 - 旋转: {rotation_weights.mean():.4f}, 质量: {quality_weights.mean():.4f}, 稳定性: {stability_weights.mean():.4f}")
        
        return combined_weights

    def _multi_method_rotation_averaging(self, qR_array, filtered_indices, weights, debug=False):
        """多方法融合的旋转平均，提高小样本的鲁棒性"""
        if len(filtered_indices) == 0:
            filtered_indices = list(range(len(qR_array)))
        
        filtered_qR = qR_array[filtered_indices]
        filtered_weights = weights[filtered_indices]
        filtered_weights = filtered_weights / np.sum(filtered_weights)  # 重新归一化
        
        quaternions = [R.from_matrix(mat).as_quat() for mat in filtered_qR]
        quaternions = np.array(quaternions)
        
        # 方法1: 加权四元数平均
        reference_quat = np.median(quaternions, axis=0)
        reference_quat /= np.linalg.norm(reference_quat)
        
        weighted_quat_1 = np.zeros(4)
        for quat, weight in zip(quaternions, filtered_weights):
            if np.dot(reference_quat, quat) < 0:
                quat = -quat
            weighted_quat_1 += weight * quat
        
        # 安全的归一化
        quat_norm = np.linalg.norm(weighted_quat_1)
        if quat_norm < 1e-8:
            weighted_quat_1 = np.array([0, 0, 0, 1])  # 单位四元数
        else:
            weighted_quat_1 /= quat_norm
        
        # 方法2: 鲁棒四元数平均（使用Huber权重）
        huber_delta = getattr(self, 'huber_delta', 0.3)
        huber_quat = self._huber_quaternion_averaging(quaternions, filtered_weights, huber_delta)
        
        # 方法3: 基于李代数的加权平均（更适合小角度变化）
        lie_quat = self._lie_algebra_rotation_averaging(filtered_qR, filtered_weights)
        
        # 融合三种方法的结果
        candidate_quats = [weighted_quat_1, huber_quat, lie_quat]
        
        # 检查并修复零范数四元数
        valid_candidate_quats = []
        for q in candidate_quats:
            if np.linalg.norm(q) < 1e-8:
                # 如果四元数范数太小，使用单位四元数
                q = np.array([0, 0, 0, 1])
            else:
                q = q / np.linalg.norm(q)  # 归一化
            valid_candidate_quats.append(q)
        
        from scipy.spatial.transform import Rotation as R_scipy
        candidate_matrices = [R_scipy.from_quat(q).as_matrix() for q in valid_candidate_quats]
        
        # 选择与大多数数据最一致的结果
        best_matrix = self._select_best_rotation(candidate_matrices, filtered_qR, filtered_weights)
        
        if debug:
            angles = [np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)) * 180 / np.pi for m in candidate_matrices]
            print(f"候选旋转角度: {angles} 度")
        
        return best_matrix

    def _huber_quaternion_averaging(self, quaternions, weights, delta=0.3):
        """使用Huber损失的鲁棒四元数平均"""
        reference_quat = np.median(quaternions, axis=0)
        reference_quat /= np.linalg.norm(reference_quat)
        
        # 迭代优化
        current_quat = reference_quat.copy()
        for _ in range(5):  # 少量迭代即可
            huber_weights = []
            for quat in quaternions:
                if np.dot(current_quat, quat) < 0:
                    quat = -quat
                dot_product = np.abs(np.dot(current_quat, quat))
                angle_dist = 2 * np.arccos(np.clip(dot_product, -1, 1))
                
                # Huber权重
                if angle_dist <= delta:
                    huber_weight = 1.0
                else:
                    huber_weight = delta / angle_dist
                huber_weights.append(huber_weight)
            
            huber_weights = np.array(huber_weights) * weights
            huber_weights /= np.sum(huber_weights)
            
            # 加权平均
            weighted_quat = np.zeros(4)
            for quat, weight in zip(quaternions, huber_weights):
                if np.dot(current_quat, quat) < 0:
                    quat = -quat
                weighted_quat += weight * quat
            
            quat_norm = np.linalg.norm(weighted_quat)
            if quat_norm < 1e-8:
                weighted_quat = np.array([0, 0, 0, 1])  # 单位四元数
            else:
                weighted_quat /= quat_norm
            
            if np.linalg.norm(weighted_quat - current_quat) < 1e-6:
                break
            current_quat = weighted_quat
        
        return current_quat

    def _lie_algebra_rotation_averaging(self, rotation_matrices, weights):
        """基于李代数的旋转平均，适合小角度变化"""
        if len(rotation_matrices) == 0:
            return np.array([0, 0, 0, 1])
        
        # 选择一个参考旋转（权重最大的）
        ref_idx = np.argmax(weights)
        R_ref = rotation_matrices[ref_idx]
        
        # 将所有旋转转换到李代数空间
        weighted_log_sum = np.zeros(3)
        total_weight = 0
        
        for R, weight in zip(rotation_matrices, weights):
            # 相对旋转
            R_rel = R @ R_ref.T
            
            # 转换到李代数（轴角表示）
            from scipy.spatial.transform import Rotation as R_scipy
            rotation_obj = R_scipy.from_matrix(R_rel)
            axis_angle = rotation_obj.as_rotvec()
            
            weighted_log_sum += weight * axis_angle
            total_weight += weight
        
        # 平均李代数向量
        avg_log = weighted_log_sum / total_weight
        
        # 转换回旋转矩阵
        from scipy.spatial.transform import Rotation as R_scipy
        avg_R_rel = R_scipy.from_rotvec(avg_log).as_matrix()
        avg_R = avg_R_rel @ R_ref
        
        # 转换为四元数
        return R_scipy.from_matrix(avg_R).as_quat()

    def _select_best_rotation(self, candidate_matrices, reference_matrices, weights):
        """选择与参考数据最一致的旋转"""
        best_score = -1
        best_matrix = candidate_matrices[0]
        
        for candidate in candidate_matrices:
            score = 0
            for ref_matrix, weight in zip(reference_matrices, weights):
                # 计算旋转角度差异
                R_diff = candidate @ ref_matrix.T
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                # 使用高斯权重
                consistency = np.exp(-angle_diff / 0.3)
                score += weight * consistency
            
            if score > best_score:
                best_score = score
                best_matrix = candidate
        
        return best_matrix

    def _enhanced_camera_center_triangulation(self, lines, filtered_indices, weights, pose_qualities, debug=False):
        """增强的相机中心三角化，结合多种信息和LUD算法"""
        if len(filtered_indices) == 0:
            if self.use_lud:
                return self.lud_camera_center_estimation(lines, weights, debug=debug)
            else:
                return self.camera_center_triangulation(lines)
        
        # 获取初始估计作为LUD的初始猜测
        try:
            initial_center = self._weighted_camera_center_triangulation(lines, filtered_indices, weights)
        except:
            initial_center = np.mean(lines[:, 0, :], axis=0)  # 简单平均作为备选
        
        candidates = []
        
        # 方法1: LUD鲁棒估计（主要方法）
        if self.use_lud:
            try:
                # 使用过滤后的数据和权重
                filtered_lines = lines[filtered_indices] if len(filtered_indices) > 0 else lines
                filtered_weights = weights[filtered_indices] if len(filtered_indices) > 0 else weights
                
                lud_center = self.lud_camera_center_estimation(
                    filtered_lines, 
                    weights=filtered_weights,
                    initial_guess=initial_center,
                    debug=debug
                )
                candidates.append(lud_center)
                
                if debug:
                    print(f"LUD估计位置: {lud_center}")
                    
            except Exception as e:
                if debug:
                    print(f"LUD方法失败: {e}")
        
        # 方法2: 基于旋转一致性的加权三角化（备选）
        try:
            center2 = self._weighted_camera_center_triangulation(lines, filtered_indices, weights)
            candidates.append(center2)
        except Exception as e:
            if debug:
                print(f"加权三角化失败: {e}")
        
        # 方法3: 基于pose质量的加权三角化
        try:
            quality_weights = pose_qualities / (np.sum(pose_qualities) + 1e-8)
            center3 = self._quality_weighted_triangulation(lines, quality_weights)
            candidates.append(center3)
        except Exception as e:
            if debug:
                print(f"质量加权三角化失败: {e}")
        
        # 方法4: RANSAC鲁棒三角化（对于小样本使用较小的阈值）
        try:
            ransac_threshold = getattr(self, 'ransac_threshold', 0.1)
            center4 = self._ransac_triangulation(lines, threshold=ransac_threshold, max_trials=min(100, len(lines) * 10))
            candidates.append(center4)
        except Exception as e:
            if debug:
                print(f"RANSAC三角化失败: {e}")
        
        # 如果没有有效候选，使用传统方法
        if not candidates:
            if debug:
                print("所有方法都失败，使用传统三角化")
            return self.camera_center_triangulation(lines)
        
        # 选择最一致的结果
        best_center = self._select_best_center(candidates, lines, weights, debug=debug)
        
        return best_center

    def _quality_weighted_triangulation(self, lines, quality_weights):
        """基于pose质量的加权三角化"""
        n = lines.shape[0]
        p = lines[:, 0, :]
        q = lines[:, 1, :]
        d = q - p
        
        # 构建加权最小二乘问题
        weighted_A = []
        weighted_b = []
        
        for i in range(n):
            weight = np.sqrt(quality_weights[i])
            
            d_i = d[i]
            p_i = p[i]
            d_norm_sq = np.sum(d_i ** 2)
            
            # 投影矩阵
            P = np.eye(3) - np.outer(d_i, d_i) / d_norm_sq
            
            # 加权矩阵块
            weighted_A.append(weight * P)
            weighted_b.append(weight * (P @ p_i))
        
        # 组合加权系统
        A = np.vstack(weighted_A)
        b = np.concatenate(weighted_b)
        
        # 求解加权最小二乘
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / (S + 1e-12))
        x = Vt.T @ (S_inv @ (U.T @ b))
        
        return x

    def _ransac_triangulation(self, lines, threshold=0.1, max_trials=100):
        """RANSAC鲁棒三角化"""
        n = lines.shape[0]
        if n < 3:
            return self.camera_center_triangulation(lines)
        
        best_inliers = 0
        best_center = None
        
        for _ in range(max_trials):
            # 随机选择最少数量的直线
            sample_size = min(3, n)
            sample_indices = np.random.choice(n, sample_size, replace=False)
            sample_lines = lines[sample_indices]
            
            # 三角化
            center = self.camera_center_triangulation(sample_lines)
            
            # 计算内点数量
            inliers = 0
            for i in range(n):
                p = lines[i, 0]
                d = lines[i, 1] - lines[i, 0]
                d = d / np.linalg.norm(d)
                
                # 点到直线的距离
                v = center - p
                dist = np.linalg.norm(v - np.dot(v, d) * d)
                
                if dist < threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_center = center
        
        if best_center is None:
            return self.camera_center_triangulation(lines)
        
        return best_center

    def _select_best_center(self, candidates, lines, weights, debug=False):
        """选择最佳的相机中心"""
        best_score = float('inf')
        best_center = candidates[0]
        
        for center in candidates:
            if center is None:
                continue
                
            # 计算加权重投影误差
            total_error = 0
            total_weight = 0
            
            for i, line in enumerate(lines):
                p = line[0]
                d = line[1] - line[0]
                d = d / (np.linalg.norm(d) + 1e-12)
                
                # 点到直线的距离
                v = center - p
                dist = np.linalg.norm(v - np.dot(v, d) * d)
                
                total_error += weights[i] * dist
                total_weight += weights[i]
            
            avg_error = total_error / (total_weight + 1e-12)
            
            if avg_error < best_score:
                best_score = avg_error
                best_center = center
        
        if debug:
            print(f"最佳相机中心的平均重投影误差: {best_score:.6f}")
        
        return best_center

    def set_small_sample_parameters(self, rotation_weight=0.5, quality_weight=0.3, stability_weight=0.2,
                                  angle_threshold=0.2, huber_delta=0.3, ransac_threshold=0.1,
                                  lud_loss_type=None, lud_huber_delta=None):
        """
        为小样本数据调整参数
        Args:
            rotation_weight: 旋转一致性权重 (0-1)
            quality_weight: pose质量权重 (0-1) 
            stability_weight: 几何稳定性权重 (0-1)
            angle_threshold: 角度权重的标准差 (弧度)
            huber_delta: Huber损失的阈值 (弧度)
            ransac_threshold: RANSAC的内点阈值 (米)
            lud_loss_type: LUD算法的损失函数类型 ("l1", "l2", "huber")
            lud_huber_delta: LUD算法的Huber损失阈值
        """
        self.rotation_weight = rotation_weight
        self.quality_weight = quality_weight  
        self.stability_weight = stability_weight
        self.angle_threshold = angle_threshold
        self.huber_delta = huber_delta
        self.ransac_threshold = ransac_threshold
        
        # 更新LUD参数
        if self.use_lud and hasattr(self, 'lud_localizer'):
            if lud_loss_type is not None:
                self.lud_localizer.loss_type = lud_loss_type
            if lud_huber_delta is not None:
                self.lud_localizer.huber_delta = lud_huber_delta
        
    def get_optimization_suggestions(self, n_pairs):
        """
        根据数据对数量提供优化建议
        """
        suggestions = []
        
        if n_pairs <= 5:
            suggestions.extend([
                "极小样本建议:",
                "- 确保每对数据的基线足够长（>0.1m）",
                "- 旋转角度适中（5-30度）",
                "- 考虑增加更多匹配对"
            ])
        elif n_pairs <= 10:
            suggestions.extend([
                "小样本优化建议:",
                "- 检查pose质量分布，移除明显的低质量数据",
                "- 如果结果不稳定，可以调整权重参数",
                "- 考虑使用debug=True查看详细信息"
            ])
            if self.use_lud:
                suggestions.extend([
                    "- LUD算法已启用，提供更好的鲁棒性",
                    "- 可尝试不同的损失函数：'l1'(最鲁棒)、'huber'(平衡)、'l2'(最快)",
                    "- 如有离群值，建议使用'l1'或调小'huber_delta'"
                ])
        else:
            suggestions.extend([
                "中等样本建议:",
                "- 当前算法应该工作良好",
                "- 可以适当放松过滤条件"
            ])
            
        return "\n".join(suggestions)
