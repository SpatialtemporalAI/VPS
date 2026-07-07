from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
from vps.utils.processing import w2c34_to_c2w44
from vps.utils.vggt_omega_preprocess import (
    VGGTOmegaPreprocessConfig,
    VGGTOmegaRefTensorCache,
    dtype_from_name,
    pad_image_tensors_to_common_size,
    preprocess_vggt_omega_images,
)


@dataclass
class _TemporalFrame:
    timestamp: float
    tensor: torch.Tensor
    image_path: Path
    pose_c2w: np.ndarray


@dataclass
class _PreparedOmegaInput:
    images: torch.Tensor
    cache_detail: str
    query_tensor: torch.Tensor
    temporal_key: Optional[tuple[str, str]]
    temporal_count: int
    temporal_poses: List[np.ndarray]


class VGGTOmegaModel(BasePoseModel):
    """VGGT-Omega inference wrapper for the decoupled pose pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config["system"]["device"])
        if self.device.type != "cuda":
            raise ValueError("VGGTOmegaModel requires CUDA because the upstream model uses CUDA autocast.")

        omega_cfg = config["pose"].get("vggt_omega", {})
        self.image_resolution = int(omega_cfg.get("image_resolution", 512))
        self.preprocess_mode = str(omega_cfg.get("preprocess_mode", "balanced"))
        self.patch_size = int(omega_cfg.get("patch_size", 16))
        self.cpu_num_threads = int(omega_cfg.get("cpu_num_threads", 8))
        if self.cpu_num_threads > 0:
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(self.cpu_num_threads)
            logging.info(
                "VGGT-Omega set torch CPU threads: previous=%d current=%d",
                previous_threads,
                torch.get_num_threads(),
            )
        self.preprocess_config = VGGTOmegaPreprocessConfig(
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        )
        self.enable_alignment = bool(omega_cfg.get("enable_alignment", False))
        self.dtype = dtype_from_name(config.get("system", {}).get("dtype", "bfloat16"))
        self.input_dtype = torch.float32
        self._ref_caches: dict[str, VGGTOmegaRefTensorCache] = {}
        self._ref_cache_lock = threading.Lock()
        temporal_cfg = omega_cfg.get("temporal_window", {})
        self.temporal_window_enabled = bool(temporal_cfg.get("enabled", False))
        self.temporal_window_max_frames = max(0, int(temporal_cfg.get("max_frames", 5)))
        self.temporal_window_max_age_seconds = max(
            0.0,
            float(temporal_cfg.get("max_age_seconds", 60.0)),
        )
        self.temporal_motion_max_frames = max(
            0,
            int(temporal_cfg.get("motion_max_frames", 3)),
        )
        self._temporal_windows: dict[tuple[str, str], list[_TemporalFrame]] = {}
        self._pending_temporal_queries: dict[tuple[str, str, str], torch.Tensor] = {}
        self._temporal_lock = threading.Lock()

        model_path = str(omega_cfg.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("pose.vggt_omega.model_path is required for VGGTOmegaModel.")

        self.model = VGGTOmega(enable_alignment=self.enable_alignment).eval()
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        ref_poses: Optional[List[np.ndarray]] = None,
        depth_paths: Optional[List[Path]] = None,
        poses_paths: Optional[List[Path]] = None,
        k_paths: Optional[List[Path]] = None,
        ref_cache_path: Optional[Path] = None,
        robot_id: Optional[str] = None,
        map_id: Optional[str] = None,
    ) -> PoseModelOutput:
        del ref_poses, depth_paths, poses_paths, k_paths

        prepare_start = time.time()
        image_paths = [Path(query_image), *[Path(path) for path in ref_images]]
        paths_ready_time = time.time()
        prepared = self._prepare_images(
            Path(query_image),
            [Path(path) for path in ref_images],
            ref_cache_path=ref_cache_path,
            robot_id=robot_id,
            map_id=map_id,
        )
        load_preprocess_time = time.time()
        images = prepared.images
        images = images.to(self.device)
        to_device_time = time.time()
        logging.info(
            "vggt_omega image prepare detail: num_images=%d temporal_frames=%d cache=%s "
            "paths=%.6fs load_preprocess=%.6fs to_device=%.6fs total=%.6fs shape=%s",
            int(images.shape[0]),
            prepared.temporal_count,
            prepared.cache_detail,
            paths_ready_time - prepare_start,
            load_preprocess_time - paths_ready_time,
            to_device_time - load_preprocess_time,
            to_device_time - prepare_start,
            tuple(images.shape),
        )

        infer_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, point_map = self._run_vggt_omega(images)
        expected_views = 1 + len(ref_images)
        temporal_pose_count = min(
            len(prepared.temporal_poses),
            max(0, extrinsic.shape[0] - expected_views),
        )
        temporal_poses = prepared.temporal_poses[:temporal_pose_count]
        intrinsic = self._keep_front_views(intrinsic, expected_views)
        depth_map = self._keep_front_views(depth_map, expected_views)
        depth_conf = self._keep_front_views(depth_conf, expected_views)
        point_map = self._keep_front_views(point_map, expected_views)
        extrinsic = w2c34_to_c2w44(extrinsic)
        logging.info("vggt_omega infer time: %.6fs", time.time() - infer_start)
        self._store_pending_temporal_query(
            prepared.temporal_key,
            prepared.query_tensor,
            Path(query_image),
        )

        return PoseModelOutput(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            point_map=point_map,
            point_conf=depth_conf,
            image_paths=image_paths,
            temporal_poses=temporal_poses,
        )

    def _get_ref_cache(self, ref_cache_path: Optional[Path]) -> Optional[VGGTOmegaRefTensorCache]:
        if ref_cache_path is None:
            return None

        cache_path = Path(ref_cache_path).expanduser().resolve(strict=False)
        cache_key = str(cache_path)
        cache = self._ref_caches.get(cache_key)
        if cache is not None:
            return cache

        with self._ref_cache_lock:
            cache = self._ref_caches.get(cache_key)
            if cache is not None:
                return cache
            cache = VGGTOmegaRefTensorCache.from_paths([cache_path])
            cache.validate_config(self.preprocess_config)
            self._ref_caches[cache_key] = cache
            logging.info(
                "Loaded VGGT-Omega map ref tensor cache: path=%s images=%d dtypes=%s",
                cache_path,
                cache.num_images,
                sorted(set(cache.storage_dtypes)),
            )
            return cache

    def _prepare_images(
        self,
        query_image: Path,
        ref_images: List[Path],
        ref_cache_path: Optional[Path],
        robot_id: Optional[str],
        map_id: Optional[str],
    ) -> _PreparedOmegaInput:
        temporal_key = self._temporal_key(robot_id, map_id)
        ref_cache = self._get_ref_cache(ref_cache_path)
        if ref_cache is None:
            images = preprocess_vggt_omega_images(
                [query_image, *ref_images],
                self.preprocess_config,
                storage_dtype=self.input_dtype,
            )
            history_tensors, history_poses = self._get_temporal_context(temporal_key)
            if history_tensors:
                images = pad_image_tensors_to_common_size(
                    [*list(images), *history_tensors],
                )
            query_tensor = images[0].detach().cpu()
            return _PreparedOmegaInput(
                images=images,
                cache_detail=f"disabled temporal={len(history_tensors)}",
                query_tensor=query_tensor,
                temporal_key=temporal_key,
                temporal_count=len(history_tensors),
                temporal_poses=history_poses,
            )

        query_start = time.time()
        query_tensor = preprocess_vggt_omega_images(
            [query_image],
            self.preprocess_config,
            storage_dtype=self.input_dtype,
        )[0]
        query_time = time.time() - query_start

        lookup_start = time.time()
        ref_tensors, missing = ref_cache.get_many(ref_images)
        lookup_time = time.time() - lookup_start
        if missing:
            fallback_start = time.time()
            images = preprocess_vggt_omega_images(
                [query_image, *ref_images],
                self.preprocess_config,
                storage_dtype=self.input_dtype,
            )
            fallback_time = time.time() - fallback_start
            logging.warning(
                "VGGT-Omega ref cache miss; fallback to full preprocess. missing=%d first_missing=%s fallback=%.6fs",
                len(missing),
                missing[0],
                fallback_time,
            )
            history_tensors, history_poses = self._get_temporal_context(temporal_key)
            if history_tensors:
                images = pad_image_tensors_to_common_size(
                    [*list(images), *history_tensors],
                )
            query_tensor = images[0].detach().cpu()
            return _PreparedOmegaInput(
                images=images,
                cache_detail=(
                    f"miss query={query_time:.6f}s lookup={lookup_time:.6f}s "
                    f"fallback={fallback_time:.6f}s temporal={len(history_tensors)}"
                ),
                query_tensor=query_tensor,
                temporal_key=temporal_key,
                temporal_count=len(history_tensors),
                temporal_poses=history_poses,
            )

        query_pad_start = time.time()
        query_tensor = self._pad_tensor_to_reference_shape(query_tensor, ref_tensors[0])
        query_pad_time = time.time() - query_pad_start

        stack_start = time.time()
        history_tensors, history_poses = self._get_temporal_context(temporal_key)
        temporal_context_time = time.time() - stack_start
        cast_start = time.time()
        tensors = [query_tensor.to(dtype=self.input_dtype)]
        tensors.extend(tensor.to(dtype=self.input_dtype) for tensor in ref_tensors)
        tensors.extend(tensor.to(dtype=self.input_dtype) for tensor in history_tensors)
        cast_time = time.time() - cast_start
        pad_stack_start = time.time()
        input_shapes = sorted({tuple(tensor.shape) for tensor in tensors})
        images = pad_image_tensors_to_common_size(tensors)
        pad_stack_time = time.time() - pad_stack_start
        stack_time = time.time() - stack_start
        query_tensor_for_history = images[0].detach().cpu()
        return _PreparedOmegaInput(
            images=images,
            cache_detail=(
                f"hit query={query_time:.6f}s lookup={lookup_time:.6f}s "
                f"query_pad={query_pad_time:.6f}s stack={stack_time:.6f}s "
                f"temporal_context={temporal_context_time:.6f}s "
                f"cast={cast_time:.6f}s pad_stack={pad_stack_time:.6f}s "
                f"temporal={len(history_tensors)} input_shapes={input_shapes}"
            ),
            query_tensor=query_tensor_for_history,
            temporal_key=temporal_key,
            temporal_count=len(history_tensors),
            temporal_poses=history_poses,
        )

    def _temporal_key(
        self,
        robot_id: Optional[str],
        map_id: Optional[str],
    ) -> Optional[tuple[str, str]]:
        if not self.temporal_window_enabled:
            return None
        if self.temporal_window_max_frames <= 0:
            return None
        if not robot_id or not map_id:
            return None
        return str(robot_id), str(map_id)

    @staticmethod
    def _pad_tensor_to_reference_shape(
        tensor: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        target_h = int(reference.shape[-2])
        target_w = int(reference.shape[-1])
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        if height == target_h and width == target_w:
            return tensor
        if height > target_h or width > target_w:
            raise ValueError(
                "Query tensor is larger than VGGT-Omega ref cache tensor. "
                f"query_shape={tuple(tensor.shape)} ref_shape={tuple(reference.shape)}"
            )

        output = torch.full(
            (*tensor.shape[:-2], target_h, target_w),
            1.0,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        top = (target_h - height) // 2
        left = (target_w - width) // 2
        output[..., top : top + height, left : left + width] = tensor
        return output

    def _get_temporal_context(
        self,
        key: Optional[tuple[str, str]],
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        if key is None:
            return [], []
        now = time.time()
        with self._temporal_lock:
            frames = self._prune_temporal_window_locked(key, now)
            return (
                # Stored temporal tensors are immutable in the prepare path; avoid
                # cloning them here because pad/stack will copy into the batch anyway.
                [frame.tensor for frame in frames],
                [np.array(frame.pose_c2w, copy=True) for frame in frames],
            )

    def _store_pending_temporal_query(
        self,
        key: Optional[tuple[str, str]],
        query_tensor: torch.Tensor,
        image_path: Path,
    ) -> None:
        if key is None:
            return
        pending_key = (*key, str(Path(image_path)))
        with self._temporal_lock:
            self._pending_temporal_queries[pending_key] = query_tensor.detach().cpu().to(dtype=self.input_dtype)

    def update_temporal_pose(
        self,
        robot_id: str,
        map_id: str,
        query_image: Path,
        pose_c2w: np.ndarray,
    ) -> None:
        key = self._temporal_key(robot_id, map_id)
        if key is None:
            return
        pending_key = (*key, str(Path(query_image)))
        now = time.time()
        with self._temporal_lock:
            query_tensor = self._pending_temporal_queries.pop(pending_key, None)
            if query_tensor is None:
                logging.warning(
                    "VGGT-Omega temporal pose update skipped: missing pending query tensor robot_id=%s map_id=%s image=%s",
                    robot_id,
                    map_id,
                    query_image,
                )
                return
            frames = self._prune_temporal_window_locked(key, now)
            frames.append(
                _TemporalFrame(
                    timestamp=now,
                    tensor=query_tensor.detach().cpu().to(dtype=self.input_dtype),
                    image_path=Path(query_image),
                    pose_c2w=np.asarray(pose_c2w, dtype=np.float32).copy(),
                )
            )
            if len(frames) > self.temporal_window_max_frames:
                del frames[: len(frames) - self.temporal_window_max_frames]
            self._temporal_windows[key] = frames
            logging.info(
                "VGGT-Omega temporal window updated: robot_id=%s map_id=%s size=%d max_frames=%d max_age=%.1fs tensor_shape=%s",
                key[0],
                key[1],
                len(frames),
                self.temporal_window_max_frames,
                self.temporal_window_max_age_seconds,
                tuple(frames[-1].tensor.shape),
            )

    def _prune_temporal_window_locked(
        self,
        key: tuple[str, str],
        now: float,
    ) -> list[_TemporalFrame]:
        frames = self._temporal_windows.get(key, [])
        if self.temporal_window_max_age_seconds > 0:
            frames = [
                frame
                for frame in frames
                if now - frame.timestamp <= self.temporal_window_max_age_seconds
            ]
        if len(frames) > self.temporal_window_max_frames:
            frames = frames[-self.temporal_window_max_frames :]
        self._temporal_windows[key] = frames
        return frames

    def _run_vggt_omega(self, images: torch.Tensor) -> tuple[np.ndarray, ...]:
        autocast_enabled = self.dtype != torch.float32
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=self.dtype, enabled=autocast_enabled):
                predictions = self.model(images)
                pred_extrinsic, pred_intrinsic = encoding_to_camera(
                    predictions["pose_enc"],
                    predictions["images"].shape[-2:],
                )

        extrinsic = self._squeeze_prediction(pred_extrinsic)
        intrinsic = self._squeeze_prediction(pred_intrinsic)
        depth_map = self._squeeze_prediction(predictions["depth"])
        depth_conf = self._squeeze_prediction(predictions["depth_conf"])
        point_map = self._depth_to_camera_point_map(depth_map, intrinsic)

        return (
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            point_map,
        )

    @staticmethod
    def _squeeze_prediction(value: torch.Tensor) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        if array.ndim >= 4 and array.shape[-1] == 1:
            array = array[..., 0]
        return array

    @staticmethod
    def _keep_front_views(array: Optional[np.ndarray], num_views: int) -> Optional[np.ndarray]:
        if array is None:
            return None
        if array.shape[0] <= num_views:
            return array
        return array[:num_views]

    @staticmethod
    def _depth_to_camera_point_map(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_map)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth shape (N,H,W), got {depth.shape}")

        num_frames, height, width = depth.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x = np.broadcast_to(x[None], (num_frames, height, width))
        y = np.broadcast_to(y[None], (num_frames, height, width))

        fx = intrinsic[:, 0, 0][:, None, None]
        fy = intrinsic[:, 1, 1][:, None, None]
        cx = intrinsic[:, 0, 2][:, None, None]
        cy = intrinsic[:, 1, 2][:, None, None]

        safe_fx = np.where(np.abs(fx) > 1e-9, fx, 1.0)
        safe_fy = np.where(np.abs(fy) > 1e-9, fy, 1.0)
        return np.stack(
            [
                (x - cx) / safe_fx * depth,
                (y - cy) / safe_fy * depth,
                depth,
            ],
            axis=-1,
        ).astype(np.float32)
