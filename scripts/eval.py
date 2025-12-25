#!/usr/bin/env python3
"""
Test vps localization functionality
"""

import sys
from pathlib import Path
import numpy as np
import os
import logging

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.utils.metric import get_translation_error, get_transl_ang_err, get_rot_err
import datetime
from vps.core import VisualPositioningSystem
import yaml
import time

# 假设vps结果和gt都为4x4的txt
def evaluate(query_dir, result_dir, gt_dir, result_txt_path):
    thresholds = [
        (1, 1),(3,3), (5, 5),(5,10),(10,10),(5,20),(10,20) ,(20, 20)
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
            r_err = get_rot_err(R_pred, R_gt)
            t_err = get_translation_error(t_pred, t_gt)
            t_angle_err = get_transl_ang_err(t_pred, t_gt)
            result_str = f"{name}: t_err={t_err:.4f}cm, r_err={r_err:.4f}deg, t_angle_err={t_angle_err:.4f}deg"
            if r_err < failure_r_thresh and t_err < failure_t_thresh:
                successful_t_errs.append(t_err)
                successful_r_errs.append(r_err)
                successful_results.append(result_str)
                for i, (r_th, t_th) in enumerate(thresholds):
                    if r_err < r_th and t_err < t_th:
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
query_dir = Path("/data/nvme0n1/phw/cambridge/Cambridge_GreatCourt/test/rgb")
vps = VisualPositioningSystem(config_path=config_path)
start_time = time.time()
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
result_dir = "/data/nvme0n1/phw/cambridge/Cambridge_GreatCourt/outputs/poses"
gt_dir = "/data/nvme0n1/phw/cambridge/Cambridge_GreatCourt/test/poses"
result_txt_path = "/data/nvme0n1/phw/cambridge/Cambridge_GreatCourt/outputs/result.txt"
evaluate(query_dir, result_dir, gt_dir, result_txt_path)



    