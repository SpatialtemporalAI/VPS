import os
import shutil
import numpy as np
import cv2

base_dir = "/home/phw/newdisk1/VPS_data/7/7scenes_source/heads/"
train_txt = os.path.join(base_dir, "TrainSplit.txt")
test_txt = os.path.join(base_dir, "TestSplit.txt")
out_dir = "/home/phw/visual-localization/VPS/datahead"

train_rgb_out = os.path.join(out_dir, "train/rgb")
test_rgb_out = os.path.join(out_dir, "test/rgb")
train_depth_out = os.path.join(out_dir, "train/depth")
test_depth_out = os.path.join(out_dir, "test/depth")
train_pose_out = os.path.join(out_dir, "train/poses")
test_pose_out = os.path.join(out_dir, "test/poses")

os.makedirs(train_rgb_out, exist_ok=True)
os.makedirs(test_rgb_out, exist_ok=True)
os.makedirs(train_depth_out, exist_ok=True)
os.makedirs(test_depth_out, exist_ok=True)
os.makedirs(train_pose_out, exist_ok=True)
os.makedirs(test_pose_out, exist_ok=True)

def read_split(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def copy_split(seqs, rgb_out, depth_out, pose_out):
    
    for seq in seqs:
        seq_dir = os.path.join(base_dir, seq)
        if not os.path.isdir(seq_dir):
            print(f"跳过不存在的目录: {seq_dir}")
            continue
        for root, _, files in os.walk(seq_dir):
            for file in files:
                src = os.path.join(root, file)
                if file.endswith("depth.png"):
                    rel = os.path.relpath(src, seq_dir)
                    new_name = f"{seq}-{rel.replace('depth.png', 'npy')}"
                    dst = os.path.join(depth_out, new_name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    depth_mm = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                    if depth_mm is None:
                        print(f"读取深度失败，跳过: {src}")
                    else:
                        depth_m = depth_mm.astype(np.float32) / 1000.0
                        np.save(dst, depth_m)
                        print(f"Saved Depth(m): {src} -> {dst}")
                    continue

                if file.endswith(".png"):
                    rel = os.path.relpath(src, seq_dir) 
                    new_name = f"{seq}-{rel.replace('color.png', 'png')}"
                    dst = os.path.join(rgb_out, new_name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)
                    print(f"Copied RGB: {src} -> {dst}")
                    
                if file.endswith("pose.txt") or file.endswith("txt"):
                    rel = os.path.relpath(src, seq_dir) 
                    new_name = f"{seq}-{rel.replace('pose.txt', 'txt')}"
                    dst = os.path.join(pose_out, new_name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)
                    print(f"Copied Pose: {src} -> {dst}")

        # for root, _, files in os.walk(pose_out):
        #     for file in files:
        #         if file.endswith("pose.txt"):
        #             old_path = os.path.join(root, file)
        #             new_name = file.replace("pose.txt", "txt")
        #             new_path = os.path.join(root, new_name)

        #             os.rename(old_path, new_path)
        #             print(f"Renamed {old_path} -> {new_path}")

# 读 split
train_seqs = read_split(train_txt)
test_seqs = read_split(test_txt)

# 分别复制 RGB、Depth 和 Pose
copy_split(train_seqs, train_rgb_out, train_depth_out, train_pose_out)
copy_split(test_seqs, test_rgb_out, test_depth_out, test_pose_out)
