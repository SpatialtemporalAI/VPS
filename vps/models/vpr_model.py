from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import time
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from hloc import extract_features, extractors
from hloc.extract_features import ImageDataset
from hloc.utils.base_model import dynamic_load


class VPRModel:
    """Global descriptor encoder.

    This class owns only the retrieval backbone and descriptor extraction.
    """

    def __init__(self, config: Dict):
        self.device = config["system"]["device"]
        self.method = config["vpr"].get("method", "megaloc")
        if self.method == "netvlad":
            self.retrieval_conf = deepcopy(extract_features.confs["netvlad"])
        elif self.method == "megaloc":
            self.retrieval_conf = deepcopy(extract_features.confs["megaloc"])
        else:
            raise ValueError(f"Unsupported VPR method: {self.method}")
        if config["vpr"].get("resize_max") is not None:
            self.retrieval_conf.setdefault("preprocessing", {})["resize_max"] = int(
                config["vpr"]["resize_max"]
            )

        self.model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])(
            self.retrieval_conf["model"]
        ).eval().to(self.device)

    def extract_global_descriptors(
        self, images: Union[str, Path, List[Union[str, Path]]], output_path: Path
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = extract_features.main(
            self.retrieval_conf,
            images,
            feature_path=output_path,
            overwrite=True,
            model=self.model,
        )
        logging.info(
            "VPR descriptor extraction detail: method=%s images=%s output=%s time=%.6fs",
            self.method,
            images,
            output_path,
            time.time() - start,
        )
        return result

    def extract_single_global_descriptor(self, image_path: Union[str, Path], output_path: Path) -> Path:
        """Fast path for online single-query VPR extraction.

        hloc.extract_features.main is optimized for batch extraction. For one
        query image it pays DataLoader worker/tqdm setup overhead every request.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_start = time.time()
        dataset_start = time.time()
        dataset = ImageDataset(
            image_path.parent,
            self.retrieval_conf["preprocessing"],
            paths=[image_path.name],
        )
        data = dataset[0]
        dataset_time = time.time() - dataset_start

        to_tensor_start = time.time()
        image = torch.from_numpy(data["image"])[None]
        image_shape = tuple(image.shape)
        image = image.to(self.device, non_blocking=True)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        to_device_time = time.time() - to_tensor_start

        forward_start = time.time()
        with torch.inference_mode():
            pred = self.model({"image": image})
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        forward_time = time.time() - forward_start

        to_cpu_start = time.time()
        pred_np = {key: value[0].detach().cpu().numpy() for key, value in pred.items()}
        pred_np["image_size"] = np.asarray(data["original_size"])
        for key, value in list(pred_np.items()):
            if value.dtype == np.float32:
                pred_np[key] = value.astype(np.float16)
        to_cpu_time = time.time() - to_cpu_start

        h5_start = time.time()
        with h5py.File(str(output_path), "w", libver="latest") as fd:
            grp = fd.create_group(image_path.name)
            for key, value in pred_np.items():
                grp.create_dataset(key, data=value)
        h5_time = time.time() - h5_start
        total_time = time.time() - total_start
        logging.info(
            "VPR single descriptor fast detail: method=%s image=%s output=%s "
            "dataset=%.6fs to_device=%.6fs forward=%.6fs to_cpu=%.6fs h5_write=%.6fs "
            "total=%.6fs image_shape=%s",
            self.method,
            image_path,
            output_path,
            dataset_time,
            to_device_time,
            forward_time,
            to_cpu_time,
            h5_time,
            total_time,
            image_shape,
        )
        return output_path
