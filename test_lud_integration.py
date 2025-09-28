#!/usr/bin/env python3
"""
测试LUD算法集成到Motion Averaging的效果
"""
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入模块
from vps.utils.motion_averaging import MotionAveraging

def create_test_data_with_outliers(n_pairs=10, outlier_ratio=0.2):
    """创建包含离群值的测试数据"""
    np.random.seed(42)
    
    # 真实查询pose
    true_query_pose = np.eye(4)
    true_query_pose[:3, :3] = np.array([
        [0.9, -0.3, 0.3],
        [0.3, 0.95, 0.1],
        [-0.3, 0.05, 0.95]
    ])
    true_query_pose[:3, 3] = [1.0, 2.0, 3.0]
    
    # 创建参考poses和相对poses
    poses_db = []
    poses_q2d = []
    
    n_outliers = int(n_pairs * outlier_ratio)
    n_inliers = n_pairs - n_outliers
    
    for i in range(n_pairs):
        # 参考pose
        ref_pose = np.eye(4)
        
        # 随机旋转
        angle = np.random.uniform(-0.3, 0.3)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        ref_pose[:3, :3] = R
        ref_pose[:3, 3] = true_query_pose[:3, 3] + np.random.randn(3) * 0.5
        
        poses_db.append(ref_pose)
        
        # 计算相对pose
        true_q2r = np.linalg.inv(ref_pose) @ true_query_pose
        
        if i < n_inliers:
            # 内点：添加小噪声
            noise_rot = np.random.randn(3) * 0.05
            noise_trans = np.random.randn(3) * 0.02
        else:
            # 离群值：添加大噪声
            noise_rot = np.random.randn(3) * 0.5
            noise_trans = np.random.randn(3) * 0.3
            print(f"添加离群值到第{i}个数据")
        
        # 旋转噪声
        K_noise = np.array([
            [0, -noise_rot[2], noise_rot[1]],
            [noise_rot[2], 0, -noise_rot[0]],
            [-noise_rot[1], noise_rot[0], 0]
        ])
        R_noise = np.eye(3) + K_noise
        
        noisy_q2r = true_q2r.copy()
        noisy_q2r[:3, :3] = R_noise @ true_q2r[:3, :3]
        noisy_q2r[:3, 3] += noise_trans
        
        # 归一化平移向量
        noisy_q2r[:3, 3] = noisy_q2r[:3, 3] / np.linalg.norm(noisy_q2r[:3, 3])
        
        poses_q2d.append(noisy_q2r)
    
    return poses_db, poses_q2d, true_query_pose

def test_lud_vs_traditional():
    """比较LUD算法与传统方法的性能"""
    print("🧪 LUD算法 vs 传统方法性能对比")
    print("=" * 60)
    
    # 创建测试数据
    poses_db, poses_q2d, true_query_pose = create_test_data_with_outliers(n_pairs=10, outlier_ratio=0.3)
    
    print(f"真实查询位置: {true_query_pose[:3, 3]}")
    print(f"真实查询旋转角度: {np.arccos((np.trace(true_query_pose[:3, :3]) - 1) / 2) * 180 / np.pi:.2f}°")
    print()
    
    # 测试传统方法
    print("📊 传统方法测试:")
    ma_traditional = MotionAveraging(use_lud=False)
    ma_traditional.set_small_sample_parameters(
        rotation_weight=0.6,
        quality_weight=0.3,
        stability_weight=0.1
    )
    
    try:
        traditional_pose = ma_traditional.motion_averaging(poses_db, poses_q2d, debug=True)
        
        pos_error_trad = np.linalg.norm(traditional_pose[:3, 3] - true_query_pose[:3, 3])
        R_error_trad = traditional_pose[:3, :3] @ true_query_pose[:3, :3].T
        angle_error_trad = np.arccos(np.clip((np.trace(R_error_trad) - 1) / 2, -1, 1)) * 180 / np.pi
        
        print(f"✅ 传统方法结果:")
        print(f"   位置误差: {pos_error_trad:.4f} 米")
        print(f"   旋转误差: {angle_error_trad:.2f} 度")
        
    except Exception as e:
        print(f"❌ 传统方法失败: {e}")
        pos_error_trad = float('inf')
        angle_error_trad = float('inf')
    
    print("\n" + "-" * 60)
    
    # 测试LUD方法
    print("📊 LUD方法测试:")
    
    for loss_type in ['huber', 'l1', 'l2']:
        print(f"\n🔧 LUD损失函数: {loss_type}")
        
        ma_lud = MotionAveraging(use_lud=True, lud_loss_type=loss_type)
        ma_lud.set_small_sample_parameters(
            rotation_weight=0.6,
            quality_weight=0.3,
            stability_weight=0.1,
            lud_loss_type=loss_type,
            lud_huber_delta=0.5
        )
        
        try:
            lud_pose = ma_lud.motion_averaging(poses_db, poses_q2d, debug=True)
            
            pos_error_lud = np.linalg.norm(lud_pose[:3, 3] - true_query_pose[:3, 3])
            R_error_lud = lud_pose[:3, :3] @ true_query_pose[:3, :3].T
            angle_error_lud = np.arccos(np.clip((np.trace(R_error_lud) - 1) / 2, -1, 1)) * 180 / np.pi
            
            print(f"✅ LUD-{loss_type}结果:")
            print(f"   位置误差: {pos_error_lud:.4f} 米")
            print(f"   旋转误差: {angle_error_lud:.2f} 度")
            
            # 比较改进
            if pos_error_lud < pos_error_trad:
                improvement = (pos_error_trad - pos_error_lud) / pos_error_trad * 100
                print(f"   🎉 位置精度提升: {improvement:.1f}%")
            
            if angle_error_lud < angle_error_trad:
                improvement = (angle_error_trad - angle_error_lud) / angle_error_trad * 100
                print(f"   🎉 旋转精度提升: {improvement:.1f}%")
                
        except Exception as e:
            print(f"❌ LUD-{loss_type}失败: {e}")
    
    print("\n" + "=" * 60)

def test_different_outlier_ratios():
    """测试不同离群值比例下的性能"""
    print("\n🔍 不同离群值比例下的鲁棒性测试")
    print("=" * 60)
    
    outlier_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    for ratio in outlier_ratios:
        print(f"\n📈 离群值比例: {ratio*100:.0f}%")
        
        poses_db, poses_q2d, true_query_pose = create_test_data_with_outliers(n_pairs=10, outlier_ratio=ratio)
        
        # 传统方法
        ma_traditional = MotionAveraging(use_lud=False)
        try:
            trad_pose = ma_traditional.motion_averaging(poses_db, poses_q2d, debug=False)
            trad_error = np.linalg.norm(trad_pose[:3, 3] - true_query_pose[:3, 3])
        except:
            trad_error = float('inf')
        
        # LUD方法
        ma_lud = MotionAveraging(use_lud=True, lud_loss_type='huber')
        try:
            lud_pose = ma_lud.motion_averaging(poses_db, poses_q2d, debug=False)
            lud_error = np.linalg.norm(lud_pose[:3, 3] - true_query_pose[:3, 3])
        except:
            lud_error = float('inf')
        
        print(f"   传统方法误差: {trad_error:.4f} 米")
        print(f"   LUD方法误差:  {lud_error:.4f} 米")
        
        if lud_error < trad_error:
            improvement = (trad_error - lud_error) / trad_error * 100
            print(f"   🎯 LUD改进: {improvement:.1f}%")
        elif lud_error > trad_error:
            degradation = (lud_error - trad_error) / trad_error * 100
            print(f"   ⚠️ LUD退化: {degradation:.1f}%")
        else:
            print(f"   ➡️ 性能相当")

def main():
    """主函数"""
    print("🚀 LUD算法集成测试")
    print("基于 'Robust camera location estimation by convex programming' 论文")
    print()
    
    try:
        # 检查依赖
        import cvxpy as cp
        print("✅ CVXPY已安装")
    except ImportError:
        print("❌ 请安装CVXPY: pip install cvxpy")
        print("   或安装完整依赖: pip install -r requirements_lud.txt")
        return
    
    # 运行测试
    test_lud_vs_traditional()
    test_different_outlier_ratios()
    
    print("\n🎊 测试完成！")
    print("\n💡 使用建议:")
    print("- 对于有离群值的数据，推荐使用LUD算法")
    print("- 'huber'损失函数在大多数情况下表现最佳")  
    print("- 'l1'损失函数对离群值最鲁棒，但可能收敛较慢")
    print("- 'l2'损失函数收敛最快，但对离群值敏感")

if __name__ == "__main__":
    main()



