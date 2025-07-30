import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, List, Dict
import os
import os.path as osp
from PIL import Image
import math
import torch
from torchvision import transforms

def load_imagesPathList_as_tensor(path_list, PIXEL_LIMIT=255000):
    """
    用于pi3的图像预处理
    直接从图片路径列表加载图片,resize到统一尺寸,转为[N, 3, H, W]的tensor
    Args:
        path_list: List[str]，图片路径列表
        PIXEL_LIMIT: int,单张图片最大像素数
    Returns:
        torch.Tensor: [N, 3, H, W]
    """
    sources = []
    for img_path in path_list:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

def generate_ref_list(
    query_img: Union[str, Path], 
    ref_dir: Union[str, Path], 
    pairs_file: Union[str, Path]) -> List[str]:
    """
    根据给定的查询图像，从配对文件中生成参考图像列表。
    """
    query_name = Path(query_img).name
    ref_list = []
    with open(pairs_file, 'r') as f:
        for line in f:
            A, ref_name = line.strip().split()
            A = Path(A).name
            if A != query_name:
                continue
            ref_name = Path(ref_name).name
            ref_image = Path(ref_dir) / "rgb" / ref_name
            if Path(ref_image).exists():
                ref_list.append(str(ref_image))
            ref_render = Path(ref_dir) / "rgb_render" / ref_name
            if Path(ref_render).exists():
                ref_list.append(str(ref_render))
    
    return ref_list

def compute_scale_factor( 
                           pred_depth: np.ndarray, 
                           gt_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> float:
    """
    计算模型预测深度和真值深度之间的尺度因子。
    
    Args:
        pred_depth: 预测的深度图 np.ndarray
        gt_depth: 真值深度图 np.ndarray
        mask: 可选的有效深度值掩码
        
    Returns:
        尺度因子
    """
    if pred_depth.shape != gt_depth.shape:
        gt_depth = cv2.resize(gt_depth, 
                                (pred_depth.shape[1], pred_depth.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    if mask is None:
        mask = (gt_depth > 1e-1) & (pred_depth > 1e-1)
    valid_pred = pred_depth[mask]
    valid_gt = gt_depth[mask]
    scale_factors = valid_gt / valid_pred
    if len(scale_factors) == 0:
        return 1.0
    else:
        return float(np.median(scale_factors))

def umeyama_alignment(src, dst):
    """
    src: Nx3 predicted points
    dst: Nx3 GT points
    Returns: s, R, t
    """
    assert src.shape == dst.shape
    n = src.shape[0]
    
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    
    cov = src_centered.T @ dst_centered / n
    
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    var_src = (src_centered ** 2).sum() / n
    s = np.sum(S) / var_src
    
    t = mu_dst - s * R @ mu_src
    
    return s, R, t
