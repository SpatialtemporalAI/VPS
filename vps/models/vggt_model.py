from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import time
import logging

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
from vps.utils.processing import w2c34_to_c2w44


class VGGTModel(BasePoseModel):
    """VGGT inference wrapper.

    This class owns only the model lifecycle and raw inference. It does not
    know about maps, motion averaging, or navigation updates.
    """

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

        self.model = VGGT()
        model_path = config["pose"]["vggt"].get("model_path")
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(url))

        self.model.eval()
        self.model = self.model.to(self.device)
        self.image_load_size = config["pose"]["vggt"]["image_size"]
        self.image_resolution_size = config["pose"]["vggt"]["image_resolution_size"]

    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        depth_paths: Optional[List[Path]] = None,
        poses_paths: Optional[List[Path]] = None,
        k_paths: Optional[List[Path]] = None,
    ) -> PoseModelOutput:
        del depth_paths, poses_paths, k_paths
        prepare_start = time.time()
        image_paths = [Path(query_image), *[Path(path) for path in ref_images]]
        paths_ready_time = time.time()
        images, original_coords = load_and_preprocess_images_square(
            image_paths, self.image_load_size
        )
        load_preprocess_time = time.time()
        images = images.to(self.device)
        to_device_time = time.time()
        logging.info(
            "vggt image prepare detail: num_images=%d paths=%.6fs load_preprocess=%.6fs "
            "to_device=%.6fs total=%.6fs",
            len(image_paths),
            paths_ready_time - prepare_start,
            load_preprocess_time - paths_ready_time,
            to_device_time - load_preprocess_time,
            to_device_time - prepare_start,
        )
        s = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, point_map, point_conf = self._run_vggt(
            images=images,
            resolution=self.image_resolution_size,
        )
        extrinsic = w2c34_to_c2w44(extrinsic)
        logging.info(f"vggt infer time: {time.time() - s}")
        return PoseModelOutput(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            point_map=point_map,
            point_conf=point_conf,
            image_paths=image_paths
        )

    def _run_vggt(self, images: torch.Tensor, resolution: int) -> tuple[np.ndarray, ...]:
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        images = F.interpolate(
            images, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images, ps_idx
            )
            point_map, point_conf = self.model.point_head(
                aggregated_tokens_list, images, ps_idx
            )

        return (
            extrinsic.squeeze(0).cpu().numpy(),
            intrinsic.squeeze(0).cpu().numpy(),
            depth_map.squeeze(0).cpu().numpy(),
            depth_conf.squeeze(0).cpu().numpy(),
            point_map.squeeze(0).cpu().numpy(),
            point_conf.squeeze(0).cpu().numpy(),
        )
