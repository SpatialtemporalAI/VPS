"""
LUD (Location estimation by convex programming) 算法实现
基于论文: "Robust camera location estimation by convex programming"

核心思想：
1. 将相机位置估计转化为凸优化问题
2. 使用L1范数或Huber损失提高鲁棒性
3. 通过ADMM或其他凸优化方法求解
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Dict
import warnings


class LUDLocalizer:
    """
    LUD相机位置估计器
    
    实现基于凸优化的鲁棒相机位置估计算法
    """
    
    def __init__(self, 
                 loss_type: str = "huber",
                 huber_delta: float = 1.0,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        """
        Args:
            loss_type: 损失函数类型 ("l1", "l2", "huber")
            huber_delta: Huber损失的阈值参数
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            verbose: 是否输出详细信息
        """
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
    def estimate_location(self, 
                         directions: np.ndarray,
                         reference_positions: np.ndarray,
                         weights: Optional[np.ndarray] = None,
                         initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        使用LUD算法估计相机位置
        
        Args:
            directions: (N, 3) 从参考位置指向查询位置的单位方向向量
            reference_positions: (N, 3) 参考相机位置
            weights: (N,) 每个观测的权重
            initial_guess: (3,) 初始位置猜测
            
        Returns:
            包含估计位置、残差、收敛信息的字典
        """
        N = directions.shape[0]
        
        if weights is None:
            weights = np.ones(N)
        else:
            weights = np.array(weights)
            
        # 归一化权重
        weights = weights / np.sum(weights) * N
        
        # 尝试不同的求解方法
        results = {}
        
        # 方法1: CVXPY凸优化求解
        try:
            result_cvx = self._solve_with_cvxpy(directions, reference_positions, weights)
            results['cvxpy'] = result_cvx
        except Exception as e:
            if self.verbose:
                print(f"CVXPY求解失败: {e}")
            results['cvxpy'] = None
            
        # 方法2: ADMM求解
        try:
            result_admm = self._solve_with_admm(directions, reference_positions, weights, initial_guess)
            results['admm'] = result_admm
        except Exception as e:
            if self.verbose:
                print(f"ADMM求解失败: {e}")
            results['admm'] = None
            
        # 方法3: 梯度下降求解
        try:
            result_gd = self._solve_with_gradient_descent(directions, reference_positions, weights, initial_guess)
            results['gradient'] = result_gd
        except Exception as e:
            if self.verbose:
                print(f"梯度下降求解失败: {e}")
            results['gradient'] = None
            
        # 选择最佳结果
        best_result = self._select_best_result(results)
        
        return best_result
    
    def _solve_with_cvxpy(self, directions: np.ndarray, reference_positions: np.ndarray, 
                         weights: np.ndarray) -> Dict:
        """使用CVXPY求解凸优化问题"""
        N = directions.shape[0]
        
        # 定义优化变量
        x = cp.Variable(3)  # 相机位置
        
        # 构建约束和目标函数
        residuals = []
        
        for i in range(N):
            d_i = directions[i]
            p_i = reference_positions[i]
            w_i = weights[i]
            
            # 残差：点到射线的距离
            # r_i = ||(x - p_i) - ((x - p_i)^T * d_i) * d_i||
            diff = x - p_i
            projection = cp.multiply(diff @ d_i, d_i)
            residual = diff - projection
            
            if self.loss_type == "l1":
                residuals.append(w_i * cp.norm(residual, 1))
            elif self.loss_type == "l2":
                residuals.append(w_i * cp.norm(residual, 2))
            elif self.loss_type == "huber":
                residuals.append(w_i * cp.huber(residual, M=self.huber_delta))
        
        # 目标函数
        objective = cp.Minimize(cp.sum(residuals))
        
        # 求解
        problem = cp.Problem(objective)
        problem.solve(solver=cp.ECOS, verbose=self.verbose)
        
        if problem.status not in ["infeasible", "unbounded"]:
            estimated_position = x.value
            final_cost = problem.value
            
            # 计算最终残差
            final_residuals = self._compute_residuals(
                estimated_position, directions, reference_positions, weights
            )
            
            return {
                'position': estimated_position,
                'cost': final_cost,
                'residuals': final_residuals,
                'status': problem.status,
                'method': 'cvxpy'
            }
        else:
            raise RuntimeError(f"CVXPY求解失败: {problem.status}")
    
    def _solve_with_admm(self, directions: np.ndarray, reference_positions: np.ndarray,
                        weights: np.ndarray, initial_guess: Optional[np.ndarray] = None,
                        rho: float = 1.0) -> Dict:
        """使用ADMM求解"""
        N = directions.shape[0]
        
        # 初始化
        if initial_guess is None:
            x = np.mean(reference_positions, axis=0)  # 简单初始化
        else:
            x = initial_guess.copy()
            
        # ADMM变量
        z = np.zeros((N, 3))  # 辅助变量
        u = np.zeros((N, 3))  # 对偶变量
        
        costs = []
        
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            
            # x-update: 二次规划子问题
            x = self._admm_x_update(x, z, u, directions, reference_positions, weights, rho)
            
            # z-update: 近端算子
            z = self._admm_z_update(x, u, weights, rho)
            
            # u-update: 对偶变量更新
            u = self._admm_u_update(x, z, u, directions, reference_positions)
            
            # 检查收敛
            primal_residual = np.linalg.norm(self._compute_primal_residual(x, z, directions, reference_positions))
            dual_residual = np.linalg.norm(rho * (z - self._admm_z_update(x_old, u, weights, rho)))
            
            cost = self._compute_cost(x, directions, reference_positions, weights)
            costs.append(cost)
            
            if self.verbose and iteration % 100 == 0:
                print(f"ADMM Iter {iteration}: cost={cost:.6f}, primal_res={primal_residual:.6f}, dual_res={dual_residual:.6f}")
            
            if primal_residual < self.tolerance and dual_residual < self.tolerance:
                if self.verbose:
                    print(f"ADMM收敛于第{iteration}次迭代")
                break
        
        final_residuals = self._compute_residuals(x, directions, reference_positions, weights)
        
        return {
            'position': x,
            'cost': costs[-1],
            'residuals': final_residuals,
            'iterations': iteration + 1,
            'costs': costs,
            'method': 'admm'
        }
    
    def _solve_with_gradient_descent(self, directions: np.ndarray, reference_positions: np.ndarray,
                                   weights: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Dict:
        """使用梯度下降求解（作为备选方案）"""
        if initial_guess is None:
            x0 = np.mean(reference_positions, axis=0)
        else:
            x0 = initial_guess.copy()
        
        def objective(x):
            return self._compute_cost(x, directions, reference_positions, weights)
        
        def gradient(x):
            return self._compute_gradient(x, directions, reference_positions, weights)
        
        # 使用scipy的优化器
        result = minimize(
            objective, x0, method='BFGS', jac=gradient,
            options={'maxiter': self.max_iterations, 'gtol': self.tolerance}
        )
        
        final_residuals = self._compute_residuals(result.x, directions, reference_positions, weights)
        
        return {
            'position': result.x,
            'cost': result.fun,
            'residuals': final_residuals,
            'iterations': result.nit,
            'success': result.success,
            'method': 'gradient_descent'
        }
    
    def _admm_x_update(self, x: np.ndarray, z: np.ndarray, u: np.ndarray,
                      directions: np.ndarray, reference_positions: np.ndarray,
                      weights: np.ndarray, rho: float) -> np.ndarray:
        """ADMM的x更新步骤"""
        N = directions.shape[0]
        
        # 构建二次规划问题: min_x ||Ax - b||^2 + rho/2 * ||Cx - d||^2
        A = np.zeros((N, 3))
        b = np.zeros(N)
        
        for i in range(N):
            d_i = directions[i]
            p_i = reference_positions[i]
            
            # 投影矩阵 P_i = I - d_i * d_i^T
            P_i = np.eye(3) - np.outer(d_i, d_i)
            A[i] = weights[i] * P_i.sum(axis=0)  # 简化处理
            b[i] = weights[i] * (P_i @ p_i).sum()
        
        # 添加ADMM项
        C = np.eye(3)
        d = (z - u).mean(axis=0)  # 简化处理
        
        # 求解 (A^T A + rho C^T C) x = A^T b + rho C^T d
        lhs = A.T @ A + rho * C.T @ C
        rhs = A.T @ b + rho * C.T @ d
        
        try:
            x_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            x_new = np.linalg.pinv(lhs) @ rhs
            
        return x_new
    
    def _admm_z_update(self, x: np.ndarray, u: np.ndarray, weights: np.ndarray, rho: float) -> np.ndarray:
        """ADMM的z更新步骤（近端算子）"""
        # 简化实现：这里应该根据具体的损失函数实现相应的近端算子
        v = x.reshape(1, -1) + u  # 广播
        
        if self.loss_type == "l1":
            # L1近端算子：软阈值
            threshold = weights.reshape(-1, 1) / rho
            z = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        elif self.loss_type == "huber":
            # Huber近端算子
            threshold = self.huber_delta
            z = np.where(np.abs(v) <= threshold, v, 
                        threshold * np.sign(v) + (v - threshold * np.sign(v)) / (1 + 1/rho))
        else:
            # L2情况下，近端算子是恒等映射
            z = v
            
        return z
    
    def _admm_u_update(self, x: np.ndarray, z: np.ndarray, u: np.ndarray,
                      directions: np.ndarray, reference_positions: np.ndarray) -> np.ndarray:
        """ADMM的u更新步骤"""
        # 计算原始残差
        primal_residual = self._compute_primal_residual(x, z, directions, reference_positions)
        return u + primal_residual.reshape(-1, 3)
    
    def _compute_primal_residual(self, x: np.ndarray, z: np.ndarray,
                                directions: np.ndarray, reference_positions: np.ndarray) -> np.ndarray:
        """计算ADMM的原始残差"""
        N = directions.shape[0]
        residuals = np.zeros((N, 3))
        
        for i in range(N):
            d_i = directions[i]
            p_i = reference_positions[i]
            
            # 点到射线的向量
            diff = x - p_i
            projection = np.dot(diff, d_i) * d_i
            residuals[i] = diff - projection
            
        return residuals - z
    
    def _compute_cost(self, x: np.ndarray, directions: np.ndarray,
                     reference_positions: np.ndarray, weights: np.ndarray) -> float:
        """计算目标函数值"""
        total_cost = 0.0
        
        for i in range(len(directions)):
            d_i = directions[i]
            p_i = reference_positions[i]
            w_i = weights[i]
            
            # 点到射线的距离
            diff = x - p_i
            projection = np.dot(diff, d_i) * d_i
            residual = diff - projection
            distance = np.linalg.norm(residual)
            
            if self.loss_type == "l1":
                total_cost += w_i * distance
            elif self.loss_type == "l2":
                total_cost += w_i * distance**2
            elif self.loss_type == "huber":
                if distance <= self.huber_delta:
                    total_cost += w_i * 0.5 * distance**2
                else:
                    total_cost += w_i * (self.huber_delta * distance - 0.5 * self.huber_delta**2)
                    
        return total_cost
    
    def _compute_gradient(self, x: np.ndarray, directions: np.ndarray,
                         reference_positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算目标函数的梯度"""
        gradient = np.zeros(3)
        
        for i in range(len(directions)):
            d_i = directions[i]
            p_i = reference_positions[i]
            w_i = weights[i]
            
            # 点到射线的向量和距离
            diff = x - p_i
            projection = np.dot(diff, d_i) * d_i
            residual = diff - projection
            distance = np.linalg.norm(residual)
            
            if distance > 1e-12:  # 避免除零
                if self.loss_type == "l1":
                    grad_contribution = w_i * residual / distance
                elif self.loss_type == "l2":
                    grad_contribution = 2 * w_i * residual
                elif self.loss_type == "huber":
                    if distance <= self.huber_delta:
                        grad_contribution = w_i * residual
                    else:
                        grad_contribution = w_i * self.huber_delta * residual / distance
                        
                # 投影到与射线垂直的空间
                P_i = np.eye(3) - np.outer(d_i, d_i)
                gradient += P_i @ grad_contribution
                
        return gradient
    
    def _compute_residuals(self, x: np.ndarray, directions: np.ndarray,
                          reference_positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算每个观测的残差"""
        residuals = []
        
        for i in range(len(directions)):
            d_i = directions[i]
            p_i = reference_positions[i]
            
            diff = x - p_i
            projection = np.dot(diff, d_i) * d_i
            residual_vec = diff - projection
            distance = np.linalg.norm(residual_vec)
            
            residuals.append(distance)
            
        return np.array(residuals)
    
    def _select_best_result(self, results: Dict) -> Dict:
        """从多个求解结果中选择最佳的"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            raise RuntimeError("所有求解方法都失败了")
        
        # 选择代价最小的结果
        best_method = min(valid_results.keys(), key=lambda k: valid_results[k]['cost'])
        best_result = valid_results[best_method]
        
        if self.verbose:
            print(f"选择了方法: {best_method}, 代价: {best_result['cost']:.6f}")
            
        return best_result


def convert_lines_to_lud_format(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将motion averaging中的lines格式转换为LUD算法需要的格式
    
    Args:
        lines: (N, 2, 3) 每条射线由两个3D点定义
        
    Returns:
        directions: (N, 3) 单位方向向量
        reference_positions: (N, 3) 参考位置
    """
    N = lines.shape[0]
    directions = np.zeros((N, 3))
    reference_positions = lines[:, 0, :]  # 射线起点作为参考位置
    
    for i in range(N):
        p = lines[i, 0]  # 起点
        q = lines[i, 1]  # 终点
        
        # 计算单位方向向量
        direction = q - p
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-12:
            directions[i] = direction / direction_norm
        else:
            # 如果方向向量为零，使用默认方向
            directions[i] = np.array([1, 0, 0])
            
    return directions, reference_positions


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    # 真实相机位置
    true_position = np.array([1.0, 2.0, 3.0])
    
    # 生成测试射线
    N = 10
    reference_positions = np.random.randn(N, 3) * 2
    
    # 从参考位置指向真实位置的方向
    directions = []
    for i in range(N):
        direction = true_position - reference_positions[i]
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)
    directions = np.array(directions)
    
    # 添加噪声
    noise_std = 0.1
    directions += np.random.randn(N, 3) * noise_std
    
    # 重新归一化
    for i in range(N):
        directions[i] = directions[i] / np.linalg.norm(directions[i])
    
    # 测试LUD算法
    localizer = LUDLocalizer(loss_type="huber", verbose=True)
    result = localizer.estimate_location(directions, reference_positions)
    
    print(f"真实位置: {true_position}")
    print(f"估计位置: {result['position']}")
    print(f"估计误差: {np.linalg.norm(result['position'] - true_position):.6f}")
    print(f"最终代价: {result['cost']:.6f}")
