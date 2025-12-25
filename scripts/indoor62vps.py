import shutil
import numpy as np
from pathlib import Path

###原始pose是w2c的  需求逆
# ============ 配置路径 ============
source_dir = Path("/ssd1/phw/indoor6/scene5/images")
ref_split = Path("/ssd1/phw/indoor6/scene5/scene5_train.txt")
test_split = Path("/ssd1/phw/indoor6/scene5/scene5_test.txt")

ref_image_dir = Path("/ssd1/phw/indoor6/scene5/train/rgb") 
ref_pose_dir = Path("/ssd1/phw/indoor6/scene5/train/poses")   
query_image_dir = Path("/ssd1/phw/indoor6/scene5/test/rgb")
query_pose_dir = Path("/ssd1/phw/indoor6/scene5/test/poses")
ref_calibration_dir = Path("/ssd1/phw/indoor6/scene5/train/calibration")
query_calibration_dir = Path("/ssd1/phw/indoor6/scene5/test/calibration")

# 创建目标目录
for d in [ref_image_dir, ref_pose_dir, query_image_dir, query_pose_dir,
          ref_calibration_dir, query_calibration_dir]:
    d.mkdir(parents=True, exist_ok=True)


def process_split(split_file, image_dir, pose_dir, calib_dir):
    with open(split_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        key = line.split(".")[0]

        # ====== 图像 ======
        src_img = source_dir / f"{key}.color.jpg"
        dst_img = image_dir / f"{key}.jpg"
        shutil.copy(src_img, dst_img)

        # ====== 位姿 ======
        src_pose = source_dir / f"{key}.pose.txt"
        dst_pose = pose_dir / f"{key}.txt"
        pose = np.loadtxt(src_pose)  # 3x4
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0, 0, 0, 1]])
        pose = np.linalg.inv(pose)
        np.savetxt(dst_pose, pose, fmt="%.6f")

        src_intr = source_dir / f"{key}.intrinsics.txt"
        dst_intr = calib_dir / f"{key}.txt"

        with open(src_intr, "r") as f:
            intr_lines = [l.strip() for l in f if l.strip()]

        fx = float(intr_lines[0].split()[2])
        with open(dst_intr, "w") as f:
            f.write(f"{fx:.6f}\n")

        print(f"Processed {key}")


# 处理 ref (train) 和 query (test)
process_split(ref_split, ref_image_dir, ref_pose_dir, ref_calibration_dir)
process_split(test_split, query_image_dir, query_pose_dir, query_calibration_dir)
