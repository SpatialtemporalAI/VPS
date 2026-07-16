from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, match_pair, numpy_image_to_torch


@dataclass
class ImageMatches:
    query_points: np.ndarray
    render_points: np.ndarray
    scores: np.ndarray


class SuperPointLightGlueMatcher:
    def __init__(
        self,
        device: str = "cuda",
        max_keypoints: int = 4096,
        detection_threshold: float = 0.0005,
    ):
        self.device = device
        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=detection_threshold,
        ).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def match(self, query_image: Path, render_rgb: np.ndarray) -> ImageMatches:
        image0 = load_image(query_image).to(self.device)
        image1 = self._rgb_to_tensor(render_rgb).to(self.device)
        feats0, feats1, matches01 = match_pair(
            self.extractor,
            self.matcher,
            image0,
            image1,
            device=self.device,
        )
        pairs, scores = self._extract_pairs_and_scores(matches01)
        if pairs.size == 0:
            return ImageMatches(
                query_points=np.empty((0, 2), dtype=np.float32),
                render_points=np.empty((0, 2), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )
        if scores.shape[0] != pairs.shape[0]:
            scores = np.ones((pairs.shape[0],), dtype=np.float32)

        keypoints0 = feats0["keypoints"].detach().cpu().numpy().astype(np.float32).reshape(-1, 2)
        keypoints1 = feats1["keypoints"].detach().cpu().numpy().astype(np.float32).reshape(-1, 2)
        in_bounds = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 0] < keypoints0.shape[0])
            & (pairs[:, 1] >= 0)
            & (pairs[:, 1] < keypoints1.shape[0])
        )
        if not np.all(in_bounds):
            pairs = pairs[in_bounds]
            scores = scores[in_bounds] if scores.shape[0] == in_bounds.shape[0] else scores
        if pairs.size == 0:
            return ImageMatches(
                query_points=np.empty((0, 2), dtype=np.float32),
                render_points=np.empty((0, 2), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )
        query_points = keypoints0[pairs[:, 0]]
        render_points = keypoints1[pairs[:, 1]]
        return ImageMatches(query_points=query_points, render_points=render_points, scores=scores)

    @staticmethod
    def _extract_pairs_and_scores(matches01: dict) -> tuple[np.ndarray, np.ndarray]:
        raw_matches = matches01.get("matches")
        raw_scores = matches01.get("scores")
        if raw_matches is not None:
            matches = raw_matches.detach().cpu().numpy()
            if matches.ndim == 2 and matches.shape[1] == 2:
                scores = (
                    raw_scores.detach().cpu().numpy().astype(np.float32)
                    if raw_scores is not None
                    else np.ones((matches.shape[0],), dtype=np.float32)
                )
                return matches.astype(np.int64), scores
            if matches.ndim == 1:
                if matches.shape[0] == 2 and np.all(matches >= 0):
                    return matches.reshape(1, 2).astype(np.int64), np.ones((1,), dtype=np.float32)
                valid = matches >= 0
                query_ids = np.nonzero(valid)[0]
                pairs = np.stack([query_ids, matches[valid]], axis=1).astype(np.int64)
                score_tensor = matches01.get("matching_scores0")
                if score_tensor is not None:
                    score_values = score_tensor.detach().cpu().numpy()
                    if score_values.ndim == 1 and score_values.shape[0] == matches.shape[0]:
                        scores = score_values[valid].astype(np.float32)
                    else:
                        scores = np.ones((pairs.shape[0],), dtype=np.float32)
                else:
                    scores = np.ones((pairs.shape[0],), dtype=np.float32)
                return pairs, scores

        raw_matches0 = matches01.get("matches0")
        if raw_matches0 is None:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
        matches0 = raw_matches0.detach().cpu().numpy()
        valid = matches0 >= 0
        query_ids = np.nonzero(valid)[0]
        pairs = np.stack([query_ids, matches0[valid]], axis=1).astype(np.int64)
        score_tensor = matches01.get("matching_scores0")
        if score_tensor is not None:
            score_values = score_tensor.detach().cpu().numpy()
            if score_values.ndim == 1 and score_values.shape[0] == matches0.shape[0]:
                scores = score_values[valid].astype(np.float32)
            else:
                scores = np.ones((pairs.shape[0],), dtype=np.float32)
        else:
            scores = np.ones((pairs.shape[0],), dtype=np.float32)
        return pairs, scores

    @staticmethod
    def _rgb_to_tensor(image: np.ndarray) -> torch.Tensor:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
        # LightGlue's numpy helper expects RGB uint8/float image.
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return numpy_image_to_torch(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
