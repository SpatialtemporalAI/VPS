from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "OmniVGGT-official"
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

from safetensors.torch import load_file as load_safetensors_file

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
from vps.utils.processing import w2c34_to_c2w44


class OmniVGGTModel(BasePoseModel):
    """OmniVGGT inference wrapper.

    Auxiliary inputs follow a narrowed contract:
    - only reference poses are consumed as camera priors
    - intrinsics are fixed and hardcoded for those reference poses
    - depth is not used, but zero placeholders are still passed to the model
      to keep the input structure aligned with the official OmniVGGT API
    """

    FIXED_INTRINSIC = np.array(
        [
            [800, 0.0, 800],
            [0.0, 800, 800],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config["system"]["device"])

        dtype_name = config["system"].get("dtype", "bfloat16")
        if dtype_name == "float16":
            self.dtype = torch.float16
        elif dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.image_load_size = int(config["pose"]["omnivggt"]["image_size"])
        self.image_resolution_size = int(config["pose"]["omnivggt"]["image_resolution_size"])
        self.model = OmniVGGT().to(self.device)
        model_path = str(config["pose"]["omnivggt"].get("model_path", "")).strip()
        if not model_path:
            raise ValueError("pose.omnivggt.model_path is required for OmniVGGTModel.")
        if model_path.endswith(".safetensors"):
            state_dict = load_safetensors_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()


    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        ref_poses: Optional[List[np.ndarray]] = None,
        depth_paths: Optional[List[Path]] = None,
        poses_paths: Optional[List[Path]] = None,
        k_paths: Optional[List[Path]] = None,
    ) -> PoseModelOutput:
        del depth_paths, poses_paths, k_paths
        prepare_start = time.time()
        input_image_paths = [Path(path) for path in ref_images] + [Path(query_image)]
        aligned_ref_poses = self._build_ref_pose_inputs(
            ref_poses=ref_poses,
            num_refs=len(ref_images),
        )
        paths_ready_time = time.time()

        (
            images,
            extrinsics,
            intrinsics,
            depthmaps,
            masks,
            depth_indices,
            camera_indices,
        ) = self._load_images_and_aux_data(
            image_paths=input_image_paths,
            ref_pose_inputs=aligned_ref_poses,
            target_size=self.image_resolution_size,
        )
        load_preprocess_time = time.time()

        images = images.to(self.device)
        extrinsics = extrinsics.to(self.device)
        intrinsics = intrinsics.to(self.device)
        depthmaps = depthmaps.to(self.device)
        masks = masks.to(self.device)
        to_device_time = time.time()

        logging.info(
            "omnivggt image prepare detail: num_images=%d paths=%.6fs load_preprocess=%.6fs "
            "to_device=%.6fs total=%.6fs",
            len(input_image_paths),
            paths_ready_time - prepare_start,
            load_preprocess_time - paths_ready_time,
            to_device_time - load_preprocess_time,
            to_device_time - prepare_start,
        )

        infer_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, point_map, point_conf = self._run_omnivggt(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depthmaps=depthmaps,
            masks=masks,
            depth_indices=depth_indices,
            camera_indices=camera_indices,
        )
        logging.info("omnivggt infer time: %.6fs", time.time() - infer_start)

        query_index = len(ref_images)
        output_order = [query_index, *range(query_index)]
        extrinsic = w2c34_to_c2w44(extrinsic[output_order])
        intrinsic = intrinsic[output_order]
        depth_map = depth_map[output_order]
        depth_conf = depth_conf[output_order]
        point_map = point_map[output_order]
        point_conf = point_conf[output_order]
        output_image_paths = [Path(query_image), *[Path(path) for path in ref_images]]

        return PoseModelOutput(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            point_map=point_map,
            point_conf=point_conf,
            image_paths=output_image_paths,
        )

    def _run_omnivggt(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depthmaps: torch.Tensor,
        masks: torch.Tensor,
        depth_indices: List[int],
        camera_indices: List[int],
    ) -> tuple[np.ndarray, ...]:
        amp_context = nullcontext()
        if self.device.type == "cuda":
            amp_context = torch.cuda.amp.autocast(dtype=self.dtype)

        with torch.no_grad():
            with amp_context:
                predictions = self.model(
                    images=images,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    depth=depthmaps,
                    mask=masks,
                    depth_gt_index=depth_indices,
                    camera_gt_index=camera_indices,
                )

            pose_enc = predictions["pose_enc"]
            pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(
                pose_enc,
                images.shape[-2:],
            )

        return (
            pred_extrinsic.squeeze(0).cpu().numpy(),
            pred_intrinsic.squeeze(0).cpu().numpy(),
            self._squeeze_prediction(predictions["depth"]),
            self._squeeze_prediction(predictions["depth_conf"]),
            self._squeeze_prediction(predictions["world_points"]),
            self._squeeze_prediction(predictions["world_points_conf"]),
        )

    @staticmethod
    def _squeeze_prediction(value: torch.Tensor) -> np.ndarray:
        array = value.detach().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        if array.ndim >= 4 and array.shape[-1] == 1:
            array = array[..., 0]
        return array

    def _load_images_and_aux_data(
        self,
        image_paths: Sequence[Path],
        ref_pose_inputs: Sequence[Optional[np.ndarray]],
        target_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        image_tensors: List[torch.Tensor] = []
        extrinsics_list: List[np.ndarray] = []
        intrinsics_list: List[np.ndarray] = []
        depths_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []
        depth_indices: List[int] = []
        camera_indices: List[int] = []

        for idx, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert("RGB")

            width, height = image.size
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            scale_x = new_width / width
            scale_y = new_height / height
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

            crop_start_y = 0
            final_height = new_height
            if new_height > target_size:
                crop_start_y = (new_height - target_size) // 2
                final_height = target_size
                image = image.crop((0, crop_start_y, new_width, crop_start_y + target_size))

            image_tensors.append(ImgNorm(image))

            # Depth priors are intentionally disabled; keep zero placeholders so the
            # OmniVGGT call shape still matches the official API contract.
            depthmap = np.zeros((final_height, new_width), dtype=np.float32)
            mask = np.zeros((final_height, new_width), dtype=np.float32)

            extrinsic = np.zeros((3, 4), dtype=np.float32)
            intrinsic = np.zeros((3, 3), dtype=np.float32)
            if ref_pose_inputs[idx] is not None:
                camera_c2w = ref_pose_inputs[idx]
                extrinsic = self._c2w_to_w2c34(camera_c2w)
                intrinsic = self.FIXED_INTRINSIC.copy()
                intrinsic[0, 0] *= scale_x
                intrinsic[1, 1] *= scale_y
                intrinsic[0, 2] *= scale_x
                intrinsic[1, 2] *= scale_y
                if new_height > target_size:
                    intrinsic[1, 2] -= crop_start_y
                camera_indices.append(idx)

            image_tensor = image_tensors[-1]
            image_tensor, pad_top = self._fit_tensor_height(image_tensor, target_size)
            image_tensors[-1] = image_tensor
            depthmap = self._fit_array_height(depthmap, target_size)
            mask = self._fit_array_height(mask, target_size)
            if intrinsic[2, 2] != 0 and pad_top != 0:
                intrinsic[1, 2] += pad_top

            depths_list.append(depthmap.astype(np.float32))
            masks_list.append(mask.astype(np.float32))
            extrinsics_list.append(extrinsic.astype(np.float32))
            intrinsics_list.append(intrinsic.astype(np.float32))

        images = torch.stack(image_tensors, dim=0)
        depthmaps = torch.from_numpy(np.array(depths_list, dtype=np.float32))[None, ..., None]
        masks = torch.from_numpy(np.array(masks_list, dtype=np.float32))[None, ...]
        extrinsics = torch.from_numpy(np.array(extrinsics_list, dtype=np.float32))[None, ...]
        intrinsics = torch.from_numpy(np.array(intrinsics_list, dtype=np.float32))[None, ...]
        return images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices

    @staticmethod
    def _build_ref_pose_inputs(
        ref_poses: Optional[Sequence[np.ndarray]],
        num_refs: int,
    ) -> List[Optional[np.ndarray]]:
        """Build OmniVGGT-aligned camera prior slots for [ref..., query]."""
        if ref_poses is None:
            return [None] * (num_refs + 1)
        if len(ref_poses) != num_refs:
            raise ValueError(
                f"Expected {num_refs} ref poses for {num_refs} ref images, but got {len(ref_poses)}."
            )
        return [np.asarray(pose, dtype=np.float32) for pose in ref_poses] + [None]

    @staticmethod
    def _fit_tensor_height(image: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, int]:
        height = int(image.shape[-2])
        if height == target_size:
            return image, 0
        if height > target_size:
            crop_top = (height - target_size) // 2
            return image[:, crop_top : crop_top + target_size, :], 0

        pad_total = target_size - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return F.pad(image, (0, 0, pad_top, pad_bottom)), pad_top

    @staticmethod
    def _fit_array_height(array: np.ndarray, target_size: int) -> np.ndarray:
        height = int(array.shape[0])
        if height == target_size:
            return array
        if height > target_size:
            crop_top = (height - target_size) // 2
            return array[crop_top : crop_top + target_size, :]

        pad_total = target_size - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return np.pad(array, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    @staticmethod
    def _c2w_to_w2c34(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (3, 4):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = matrix
        elif matrix.shape == (4, 4):
            c2w = matrix
        else:
            raise ValueError(f"Unsupported pose shape: {matrix.shape}")
        w2c = np.linalg.inv(c2w)
        return w2c[:3, :4].astype(np.float32)
