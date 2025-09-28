#!/usr/bin/env python3
"""
相对深度导航演示脚本
展示如何基于VGGT/MapAnything的归一化深度输出进行局部导航
"""

import sys
import os
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import logging

from vps.navigation.relative_depth_navigator import RelativeDepthNavigator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_simulated_scenarios():
    """创建不同的模拟场景用于测试"""
    
    scenarios = {}
    H, W = 240, 320
    
    # 场景1: 前方有障碍物，左侧空旷
    depth1 = np.random.uniform(0.6, 0.9, (H, W))  # 背景
    depth1[H//2:, W//3:2*W//3] = np.random.uniform(0.1, 0.25, (H//2, W//3))  # 前方障碍物
    depth1[:, :W//4] = np.random.uniform(0.7, 0.95, (H, W//4))  # 左侧空旷
    scenarios['obstacle_front_left_clear'] = {
        'depth': depth1,
        'description': '前方有障碍物，左侧空旷'
    }
    
    # 场景2: 左右都有障碍物，前方相对安全
    depth2 = np.random.uniform(0.5, 0.7, (H, W))  # 背景
    depth2[:, :W//4] = np.random.uniform(0.15, 0.3, (H, W//4))  # 左侧障碍物
    depth2[:, 3*W//4:] = np.random.uniform(0.15, 0.3, (H, 3*W//4))  # 右侧障碍物
    depth2[H//2:, W//3:2*W//3] = np.random.uniform(0.6, 0.8, (H//2, W//3))  # 前方相对安全
    scenarios['sides_blocked_front_clear'] = {
        'depth': depth2,
        'description': '两侧有障碍物，前方相对安全'
    }
    
    # 场景3: 到处都是障碍物（困难场景）
    depth3 = np.random.uniform(0.1, 0.35, (H, W))  # 整体都是障碍物
    depth3[H//4:H//2, W//4:3*W//4] = np.random.uniform(0.4, 0.6, (H//4, W//2))  # 中上部分稍微远一些
    scenarios['crowded_environment'] = {
        'depth': depth3,
        'description': '拥挤环境，到处都是障碍物'
    }
    
    # 场景4: 空旷环境
    depth4 = np.random.uniform(0.7, 0.95, (H, W))  # 整体空旷
    depth4[H//3:2*H//3, 2*W//3:] = np.random.uniform(0.4, 0.6, (H//3, W//3))  # 右侧有一些远距离物体
    scenarios['open_space'] = {
        'depth': depth4,
        'description': '空旷环境'
    }
    
    return scenarios

def run_scenario_test(navigator, scenario_name, scenario_data):
    """运行单个场景测试"""
    
    print(f"\n=== 场景测试: {scenario_name} ===")
    print(f"场景描述: {scenario_data['description']}")
    
    depth_map = scenario_data['depth']
    
    # 分析深度区域
    region_analysis = navigator.analyze_depth_regions(depth_map)
    print("\n区域分析结果:")
    for region, info in region_analysis.items():
        print(f"  {region}区域:")
        print(f"    平均深度: {info['mean_depth']:.3f}")
        print(f"    最小深度: {info['min_depth']:.3f}")
        print(f"    障碍物比例: {info['obstacle_ratio']:.3f}")
        print(f"    是否安全: {info['is_safe']}")
    
    # 检测前方障碍物
    obstacle_info = navigator.detect_obstacles_in_path(depth_map)
    print(f"\n前方路径分析:")
    print(f"  近距离障碍物比例: {obstacle_info['near_obstacle_ratio']:.3f}")
    print(f"  中等距离障碍物比例: {obstacle_info['mid_obstacle_ratio']:.3f}")
    print(f"  平均前方深度: {obstacle_info['avg_front_depth']:.3f}")
    print(f"  路径是否清晰: {obstacle_info['is_path_clear']}")
    print(f"  是否需要紧急停止: {obstacle_info['emergency_stop']}")
    
    # 生成导航命令
    nav_command = navigator.generate_navigation_command(depth_map)
    print(f"\n导航命令:")
    print(f"  动作: {nav_command['action']}")
    print(f"  线速度: {nav_command['linear_velocity']:.2f}")
    print(f"  角速度: {nav_command['angular_velocity']:.2f}")
    print(f"  置信度: {nav_command['confidence']:.2f}")
    print(f"  决策原因: {nav_command['reason']}")
    
    return nav_command, region_analysis, obstacle_info

def visualize_all_scenarios(scenarios, navigator, save_dir='output'):
    """可视化所有场景的分析结果"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, len(scenarios), figsize=(5*len(scenarios), 10))
    if len(scenarios) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        depth_map = scenario_data['depth']
        nav_command = navigator.generate_navigation_command(depth_map)
        obstacle_mask = navigator.create_obstacle_mask(depth_map)
        
        # 上排：原始深度图
        axes[0, i].imshow(depth_map, cmap='viridis')
        axes[0, i].set_title(f'{scenario_data["description"]}\n深度图')
        axes[0, i].axis('off')
        
        # 下排：障碍物掩码 + 导航命令
        axes[1, i].imshow(obstacle_mask, cmap='Reds', alpha=0.7)
        axes[1, i].imshow(depth_map, cmap='viridis', alpha=0.3)
        
        # 添加导航命令文本
        command_text = f"动作: {nav_command['action']}\n"
        command_text += f"线速度: {nav_command['linear_velocity']:.2f}\n"
        command_text += f"角速度: {nav_command['angular_velocity']:.2f}"
        
        axes[1, i].text(0.02, 0.98, command_text, 
                       transform=axes[1, i].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
        
        axes[1, i].set_title('障碍物检测 + 导航命令')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'all_scenarios_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可视化结果已保存至: {save_dir / 'all_scenarios_analysis.png'}")

def test_real_mapanything_integration():
    """测试与真实MapAnything输出的集成"""
    
    print("\n=== 真实MapAnything集成测试 ===")
    
    # 模拟MapAnything的输出格式
    # 实际使用中，这些数据来自MapAnything模型的predictions
    
    # 模拟归一化深度图（MapAnything输出的depth_z通常已经归一化）
    simulated_mapanything_depth = np.random.uniform(0.0, 1.0, (256, 256))
    
    # 添加一些结构化的障碍物模式
    H, W = simulated_mapanything_depth.shape
    
    # 前方墙壁
    simulated_mapanything_depth[H//2:, W//3:2*W//3] = np.random.uniform(0.05, 0.2, (H//2, W//3))
    
    # 左侧有一些远距离物体
    simulated_mapanything_depth[:H//2, :W//4] = np.random.uniform(0.6, 0.8, (H//2, W//4))
    
    # 右侧相对空旷
    simulated_mapanything_depth[:, 3*W//4:] = np.random.uniform(0.8, 0.95, (H, W//4))
    
    # 初始化导航器
    navigator = RelativeDepthNavigator({
        'obstacle_threshold': 0.3,  # 适应MapAnything的输出范围
        'near_threshold': 0.1,
    })
    
    # 执行导航分析
    nav_command = navigator.generate_navigation_command(simulated_mapanything_depth)
    
    print("MapAnything深度图分析结果:")
    print(f"  深度图形状: {simulated_mapanything_depth.shape}")
    print(f"  深度值范围: [{simulated_mapanything_depth.min():.3f}, {simulated_mapanything_depth.max():.3f}]")
    print(f"  导航决策: {nav_command['action']}")
    print(f"  速度命令: v={nav_command['linear_velocity']:.2f}, ω={nav_command['angular_velocity']:.2f}")
    print(f"  置信度: {nav_command['confidence']:.2f}")
    
    # 可视化
    navigator.visualize_navigation_analysis(
        simulated_mapanything_depth, 
        nav_command, 
        'output/mapanything_integration_test.png'
    )

def main():
    parser = argparse.ArgumentParser(description='相对深度导航演示')
    parser.add_argument('--test-scenarios', action='store_true', 
                       help='运行模拟场景测试')
    parser.add_argument('--test-integration', action='store_true',
                       help='测试MapAnything集成')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化结果')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 初始化导航器
    navigator_config = {
        'obstacle_threshold': 0.4,   # 障碍物深度阈值
        'near_threshold': 0.15,      # 近距离阈值
        'far_threshold': 0.8,        # 远距离阈值
        'safe_zone_width': 0.3,      # 安全区域宽度
    }
    
    navigator = RelativeDepthNavigator(navigator_config)
    
    print("=== 相对深度导航系统演示 ===")
    print("核心原理:")
    print("1. 使用VGGT/MapAnything输出的归一化深度图")
    print("2. 设定相对深度阈值来区分障碍物和自由空间")
    print("3. 基于区域分析生成导航命令")
    print("4. 无需真实米制尺度，完全基于相对深度关系")
    
    if args.test_scenarios or not any([args.test_scenarios, args.test_integration]):
        # 创建并测试模拟场景
        scenarios = create_simulated_scenarios()
        
        results = {}
        for scenario_name, scenario_data in scenarios.items():
            nav_command, region_analysis, obstacle_info = run_scenario_test(
                navigator, scenario_name, scenario_data
            )
            results[scenario_name] = {
                'nav_command': nav_command,
                'region_analysis': region_analysis,
                'obstacle_info': obstacle_info
            }
        
        if args.visualize:
            visualize_all_scenarios(scenarios, navigator, args.output)
    
    if args.test_integration:
        # 测试MapAnything集成
        test_real_mapanything_integration()
    
    print(f"\n演示完成！输出文件保存在: {args.output}")
    print("\n关键优势:")
    print("✅ 无需尺度恢复 - 直接使用相对深度")
    print("✅ 实时性好 - 简单的阈值和区域分析")
    print("✅ 鲁棒性强 - 不依赖绝对尺度的准确性")
    print("✅ 易于调试 - 可视化分析过程")

if __name__ == '__main__':
    main()
