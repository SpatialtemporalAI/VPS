import os
import json
import numpy as np
import shutil

json_path = "/ssd1/phw/稀创扫描-后处理原数据/transforms.json"
folder1 = "/ssd1/phw/稀创扫描-后处理原数据/undistort/left"
folder2 = "/ssd1/phw/稀创扫描-后处理原数据/undistort/right"
output_dir = "/ssd1/phw/xichuang/ref"

rgb_dir = os.path.join(output_dir, "rgb")
poses_dir = os.path.join(output_dir, "poses")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(poses_dir, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

frames = data["frames"]

for idx, frame in enumerate(frames):

    T = np.array(frame["transform_matrix"], dtype=float)
    R = T[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[:3, :3] = R

    pose_path = os.path.join(poses_dir, f"{idx}.txt")
    np.savetxt(pose_path, T, fmt="%.8f")
    print(f"Saved pose -> {pose_path}")

    ts = str(frame["timestamp"])
    found = False
    for folder in [folder1, folder2]:
        img_path = os.path.join(folder, f"{ts}.png")
        if os.path.exists(img_path):
            new_path = os.path.join(rgb_dir, f"{idx}.png")
            shutil.copy(img_path, new_path)
            print(f"Found {img_path}, saved as {new_path}")
            found = True
            break
    if not found:
        print(f"Warning: timestamp {ts}.png not found in both folders")

print(f"\n处理完成!结果已保存到 {output_dir}/rgb 和 {output_dir}/poses")
