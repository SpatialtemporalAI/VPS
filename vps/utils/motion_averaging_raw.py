import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation as R
# from pdb import set_trace as bb


class MotionAveraging:
    def __init__(self):
        super(MotionAveraging, self).__init__()

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



    def motion_averaging(self, poses_db, poses_q2d): 
        """
        Args:
            poses_db (List): list of (4, 4) absolute poses that transform points from camera to world
            poses_q2d (List): list of (4, 4) relative poses that transform query to dbs
        Returns:
            Rt (np.ndarray): shape of (4, 4), absolute pose of query 
        """
        assert len(poses_db) == len(poses_q2d)
        qR = []
        lines = []
        for pid in range(len(poses_db)):
            pose_q = poses_db[pid] @ poses_q2d[pid]
            qR.append(pose_q[0:3,0:3])
            p_beg = poses_db[pid][0:3,3]
            p_end = (poses_db[pid] @ poses_q2d[pid][0:4,3])[0:3]
            endpoints = np.concatenate((p_beg[None,...], p_end[None,...]), axis=0)
            lines.append(endpoints)
        
        # 获取旋转平均化的过滤结果和权重
        qR_array = np.array(qR)
        filtered_indices, weights = self._get_rotation_filtering_and_weights(qR_array)
        
        # 使用过滤后的旋转进行平均
        filtered_qR = qR_array[filtered_indices]
        avg_qR = self._weighted_rotation_averaging(filtered_qR, weights[filtered_indices])
        
        # 基于旋转一致性的鲁棒加权相机中心三角化（IRLS）
        cam_cen = self._robust_weighted_camera_center_triangulation(
            np.array(lines), filtered_indices, weights
        )
        
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
        
        weighted_quat /= np.linalg.norm(weighted_quat)
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



