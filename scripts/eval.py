#!/usr/bin/env python3
"""
Test vps localization functionality
"""

import sys
from pathlib import Path
import numpy as np
import os
from analyze_translation_error import main as analyze_translation_error
import logging
import datetime
# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.core import VisualPositioningSystem
import yaml
import time
def rotation_error(R1, R2):
    # R1, R2: 3x3旋转矩阵
    R = R1 @ R2.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)

def translation_error(t1, t2):
    # t1, t2: 3维平移向量
    return np.linalg.norm(t1 - t2) * 100  # m->cm
# 假设vps结果和gt都为4x4的txt
def evaluate(query_dir, result_dir, gt_dir, result_txt_path):
    thresholds = [
        (1, 1),(3,3), (5, 5),(5,10),(5,15),(3,20),(3,30) ,(5, 10), (5, 15)
    ]
    # 失败指标：旋转误差大于5度或平移误差大于20cm
    failure_r_thresh = 1000000000000000000
    failure_t_thresh = 1000000000000000000

    counts = [0] * len(thresholds)
    total = 0
    
    successful_results = []
    failed_results = []
    
    successful_t_errs = []
    successful_r_errs = []

    for ext in ["*.jpg", "*.png"]:
        for q in Path(query_dir).glob(ext):
            name = q.stem
            pred_path = Path(result_dir) / f"{name}.txt"
            gt_path = Path(gt_dir) / f"{name}.txt"
            if not pred_path.exists() or not gt_path.exists():
                continue
            
            total += 1
            pred = np.loadtxt(pred_path)
            gt = np.loadtxt(gt_path)
            R_pred, t_pred = pred[:3, :3], pred[:3, 3]
            R_gt, t_gt = gt[:3, :3], gt[:3, 3]
            r_err = rotation_error(R_pred, R_gt)
            t_err = translation_error(t_pred, t_gt)
            angle_deg, scale, best_terror = analyze_translation_error(gt_path, pred_path)
            result_str = f"{name}: t_err={t_err:.4f}cm, r_err={r_err:.4f}deg, angle_deg={angle_deg:.4f}deg, scale={scale:.4f}, best_error={best_terror:.4f}cm"
            if r_err < failure_r_thresh and t_err < failure_t_thresh:
                successful_t_errs.append(t_err)
                successful_r_errs.append(r_err)
                successful_results.append(result_str)
                for i, (r_th, t_th) in enumerate(thresholds):
                    if r_err < r_th and best_terror < t_th:
                        counts[i] += 1
            else:
                failed_results.append(result_str)

    num_successful = len(successful_results)
    success_rate = (num_successful / total * 100) if total > 0 else 0

    with open(result_txt_path, 'w') as f:
        f.write(f"总查询数: {total}\n")
        f.write(f"成功本地化数 (R < {failure_r_thresh}度, t < {failure_t_thresh}cm): {num_successful}\n")
        f.write(f"成功率: {success_rate:.2f}%\n")
        f.write("\n" + "="*30 + "\n\n")

        f.write("在成功案例基础上，不同精度下的占比:\n")
        for i, (r_th, t_th) in enumerate(thresholds):
            percent = (counts[i] / num_successful * 100) if num_successful > 0 else 0
            f.write(f"<{r_th}度, <{t_th}cm: {percent:.2f}% ({counts[i]}/{num_successful})\n")
        
        if num_successful > 0:
            f.write("\n成功案例的误差统计:\n")
            f.write(f"平移误差 (cm): 平均={np.mean(successful_t_errs):.4f}, 中位数={np.median(successful_t_errs):.4f}, 最大值={np.max(successful_t_errs):.4f}\n")
            f.write(f"旋转误差 (deg):  平均={np.mean(successful_r_errs):.4f}, 中位数={np.median(successful_r_errs):.4f}, 最大值={np.max(successful_r_errs):.4f}\n")
            f.write("\n" + "="*30 + "\n\n")
            f.write("成功的案例列表:\n")
            for line in successful_results:
                f.write(line + '\n')

        if failed_results:
            f.write("\n" + "="*30 + "\n\n")
            f.write("失败的案例列表:\n")
            for line in failed_results:
                f.write(line + '\n')


# # Load configuration
config_path = "configs/default.yaml"
os.makedirs('../log',exist_ok=True)
log_filename = datetime.datetime.now().strftime("log/log_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize VPS
vps = VisualPositioningSystem(config_path=config_path)
start_time = time.time()
query_dir = Path("/home/phw/visual-localization/VPS/outputs/frames727")
for ext in ["*.jpg", "*.png"]:
    for query_image in sorted(query_dir.glob(ext)):
        a = query_image.stem
        query_depth = os.path.join(query_dir, f"{a}.npy")
        if os.path.exists(query_depth):
            query_depth = query_depth
        else:
            query_depth = None
        vps.localize(query_image,query_depth=query_depth)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
result_dir = "/home/phw/visual-localization/VPS/data/outputs/poses"
gt_dir = "/home/phw/visual-localization/VPS/data/ref/poses"
result_txt_path = "/home/phw/visual-localization/VPS/data/outputs/result.txt"
evaluate(query_dir, result_dir, gt_dir, result_txt_path)



    