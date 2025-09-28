import torch
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path

class RelativeDepthNavigator:
    """
    基于相对深度的局部导航器
    
    核心思想：
    1. 不需要真实米制尺度，只需要相对深度关系
    2. 将归一化深度图直接用于障碍物检测
    3. 基于相对深度阈值进行导航决策
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
            
        # 相对深度阈值配置
        self.near_threshold = config.get('near_threshold', 0.1)      # 近距离阈值（归一化深度）
        self.far_threshold = config.get('far_threshold', 0.8)        # 远距离阈值（归一化深度）
        self.obstacle_threshold = config.get('obstacle_threshold', 0.3)  # 障碍物阈值
        
        # 导航参数
        self.safe_zone_width = config.get('safe_zone_width', 0.3)    # 安全区域宽度比例
        self.turn_angle_step = config.get('turn_angle_step', 15)     # 转向角度步长（度）
        
        # 图像分割参数
        self.image_regions = {
            'left': [0, 0.33],      # 左侧区域
            'center': [0.33, 0.67], # 中央区域  
            'right': [0.67, 1.0]    # 右侧区域
        }
        
        logging.info("相对深度导航器初始化完成")
    
    def analyze_depth_regions(self, depth_map: np.ndarray) -> Dict[str, Dict]:
        """
        分析深度图的不同区域
        
        Args:
            depth_map: 归一化深度图 [H, W]，值域通常在[0, 1]
            
        Returns:
            各区域的分析结果
        """
        H, W = depth_map.shape
        results = {}
        
        for region_name, (start_ratio, end_ratio) in self.image_regions.items():
            # 计算区域边界
            start_col = int(W * start_ratio)
            end_col = int(W * end_ratio)
            
            # 提取区域深度
            region_depth = depth_map[:, start_col:end_col]
            
            # 过滤有效深度值
            valid_mask = region_depth > 0.01  # 过滤掉无效深度
            valid_depths = region_depth[valid_mask]
            
            if len(valid_depths) == 0:
                results[region_name] = {
                    'mean_depth': 1.0,
                    'min_depth': 1.0,
                    'obstacle_ratio': 0.0,
                    'is_safe': False
                }
                continue
            
            # 统计分析
            mean_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            
            # 障碍物比例（近距离像素的比例）
            obstacle_pixels = np.sum(valid_depths < self.obstacle_threshold)
            obstacle_ratio = obstacle_pixels / len(valid_depths)
            
            # 安全性评估
            is_safe = (min_depth > self.near_threshold and 
                      obstacle_ratio < 0.2 and
                      mean_depth > self.obstacle_threshold)
            
            results[region_name] = {
                'mean_depth': float(mean_depth),
                'min_depth': float(min_depth),
                'obstacle_ratio': float(obstacle_ratio),
                'is_safe': is_safe,
                'pixel_count': len(valid_depths)
            }
        
        return results
    
    def detect_obstacles_in_path(self, depth_map: np.ndarray) -> Dict:
        """
        检测路径中的障碍物
        
        Args:
            depth_map: 归一化深度图
            
        Returns:
            障碍物检测结果
        """
        H, W = depth_map.shape
        
        # 定义前方关键区域（下半部分，中央区域）
        front_region = depth_map[H//2:, W//3:2*W//3]
        
        # 检测近距离障碍物
        near_obstacles = front_region < self.near_threshold
        near_obstacle_ratio = np.mean(near_obstacles)
        
        # 检测中等距离障碍物
        mid_obstacles = (front_region >= self.near_threshold) & (front_region < self.obstacle_threshold)
        mid_obstacle_ratio = np.mean(mid_obstacles)
        
        # 计算平均前方深度
        valid_front = front_region[front_region > 0.01]
        avg_front_depth = np.mean(valid_front) if len(valid_front) > 0 else 1.0
        
        return {
            'near_obstacle_ratio': float(near_obstacle_ratio),
            'mid_obstacle_ratio': float(mid_obstacle_ratio),
            'avg_front_depth': float(avg_front_depth),
            'is_path_clear': near_obstacle_ratio < 0.1 and avg_front_depth > self.obstacle_threshold,
            'emergency_stop': near_obstacle_ratio > 0.3
        }
    
    def generate_navigation_command(self, depth_map: np.ndarray) -> Dict:
        """
        基于相对深度生成导航命令
        
        Args:
            depth_map: 归一化深度图
            
        Returns:
            导航命令字典
        """
        # 分析各区域
        region_analysis = self.analyze_depth_regions(depth_map)
        
        # 检测前方障碍物
        obstacle_info = self.detect_obstacles_in_path(depth_map)
        
        # 初始化命令
        command = {
            'linear_velocity': 0.0,      # 线速度 (相对值)
            'angular_velocity': 0.0,     # 角速度 (相对值)
            'action': 'stop',           # 动作类型
            'confidence': 0.0,          # 置信度
            'reason': ''                # 决策原因
        }
        
        # 紧急停止检查
        if obstacle_info['emergency_stop']:
            command.update({
                'action': 'emergency_stop',
                'reason': '检测到近距离障碍物',
                'confidence': 0.9
            })
            return command
        
        # 决策逻辑
        left_safe = region_analysis['left']['is_safe']
        center_safe = region_analysis['center']['is_safe']
        right_safe = region_analysis['right']['is_safe']
        
        if center_safe and obstacle_info['is_path_clear']:
            # 前方安全，直行
            command.update({
                'linear_velocity': self._calculate_safe_speed(obstacle_info['avg_front_depth']),
                'angular_velocity': 0.0,
                'action': 'forward',
                'reason': '前方路径清晰',
                'confidence': 0.8
            })
            
        elif left_safe and not right_safe:
            # 左侧安全，左转
            command.update({
                'linear_velocity': 0.2,  # 减速转弯
                'angular_velocity': 0.3,  # 左转
                'action': 'turn_left',
                'reason': '左侧区域更安全',
                'confidence': 0.7
            })
            
        elif right_safe and not left_safe:
            # 右侧安全，右转
            command.update({
                'linear_velocity': 0.2,
                'angular_velocity': -0.3,  # 右转
                'action': 'turn_right', 
                'reason': '右侧区域更安全',
                'confidence': 0.7
            })
            
        elif left_safe and right_safe:
            # 两侧都安全，选择障碍物较少的一侧
            left_obstacles = region_analysis['left']['obstacle_ratio']
            right_obstacles = region_analysis['right']['obstacle_ratio']
            
            if left_obstacles < right_obstacles:
                command.update({
                    'linear_velocity': 0.3,
                    'angular_velocity': 0.2,
                    'action': 'turn_left',
                    'reason': '左侧障碍物更少',
                    'confidence': 0.6
                })
            else:
                command.update({
                    'linear_velocity': 0.3,
                    'angular_velocity': -0.2,
                    'action': 'turn_right',
                    'reason': '右侧障碍物更少', 
                    'confidence': 0.6
                })
        else:
            # 都不安全，后退或大角度转向
            command.update({
                'linear_velocity': -0.1,  # 缓慢后退
                'angular_velocity': 0.5,  # 大角度转向
                'action': 'backup_and_turn',
                'reason': '周围都有障碍物，尝试后退转向',
                'confidence': 0.3
            })
        
        return command
    
    def _calculate_safe_speed(self, avg_depth: float) -> float:
        """
        根据平均深度计算安全速度
        
        Args:
            avg_depth: 平均深度（归一化值）
            
        Returns:
            安全线速度
        """
        if avg_depth > 0.7:
            return 0.5  # 远距离，正常速度
        elif avg_depth > 0.5:
            return 0.3  # 中等距离，中等速度
        elif avg_depth > 0.3:
            return 0.1  # 近距离，慢速
        else:
            return 0.0  # 很近，停止
    
    def create_obstacle_mask(self, depth_map: np.ndarray) -> np.ndarray:
        """
        创建障碍物掩码
        
        Args:
            depth_map: 归一化深度图
            
        Returns:
            障碍物掩码 (1=障碍物, 0=自由空间)
        """
        # 基于相对深度阈值创建障碍物掩码
        obstacle_mask = (depth_map > 0.01) & (depth_map < self.obstacle_threshold)
        
        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mask = cv2.morphologyEx(
            obstacle_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        
        return obstacle_mask.astype(bool)
    
    def visualize_navigation_analysis(self, depth_map: np.ndarray, command: Dict, save_path: str = None):
        """
        可视化导航分析结果
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 原始深度图
        axes[0, 0].imshow(depth_map, cmap='viridis')
        axes[0, 0].set_title('原始深度图 (归一化)')
        axes[0, 0].axis('off')
        
        # 2. 障碍物掩码
        obstacle_mask = self.create_obstacle_mask(depth_map)
        axes[0, 1].imshow(obstacle_mask, cmap='Reds')
        axes[0, 1].set_title('障碍物掩码')
        axes[0, 1].axis('off')
        
        # 3. 区域分析
        H, W = depth_map.shape
        region_viz = np.zeros_like(depth_map)
        
        # 标记不同区域
        region_viz[:, :int(W*0.33)] = 1  # 左侧
        region_viz[:, int(W*0.33):int(W*0.67)] = 2  # 中央
        region_viz[:, int(W*0.67):] = 3  # 右侧
        
        axes[1, 0].imshow(region_viz, cmap='Set3', alpha=0.7)
        axes[1, 0].imshow(depth_map, cmap='viridis', alpha=0.5)
        axes[1, 0].set_title('区域分析')
        axes[1, 0].axis('off')
        
        # 4. 导航命令可视化
        axes[1, 1].text(0.1, 0.9, f"动作: {command['action']}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f"线速度: {command['linear_velocity']:.2f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"角速度: {command['angular_velocity']:.2f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"置信度: {command['confidence']:.2f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"原因: {command['reason']}", fontsize=10, transform=axes[1, 1].transAxes, wrap=True)
        axes[1, 1].set_title('导航命令')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"导航分析可视化保存至: {save_path}")
        
        plt.show()


def demo_relative_depth_navigation():
    """演示相对深度导航"""
    
    # 创建模拟的归一化深度图
    H, W = 240, 320
    
    # 模拟场景1：前方有障碍物，左侧较空旷
    depth_map1 = np.random.uniform(0.6, 0.9, (H, W))  # 背景
    depth_map1[H//2:, W//3:2*W//3] = np.random.uniform(0.1, 0.3, (H//2, W//3))  # 前方障碍物
    depth_map1[:, :W//4] = np.random.uniform(0.7, 0.9, (H, W//4))  # 左侧空旷
    
    # 初始化导航器
    navigator = RelativeDepthNavigator({
        'obstacle_threshold': 0.4,
        'near_threshold': 0.15,
    })
    
    # 生成导航命令
    command = navigator.generate_navigation_command(depth_map1)
    
    print("=== 相对深度导航演示 ===")
    print(f"动作: {command['action']}")
    print(f"线速度: {command['linear_velocity']:.2f}")
    print(f"角速度: {command['angular_velocity']:.2f}")
    print(f"置信度: {command['confidence']:.2f}")
    print(f"决策原因: {command['reason']}")
    
    # 可视化
    navigator.visualize_navigation_analysis(depth_map1, command, 'navigation_demo.png')


if __name__ == "__main__":
    demo_relative_depth_navigation()
