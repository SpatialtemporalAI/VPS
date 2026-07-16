from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gsplat.rendering import rasterization
from plyfile import PlyData


@dataclass
class GaussianCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_plane: float = 0.01
    far_plane: float = 1.0e4

    @classmethod
    def from_mapping(
        cls,
        data: dict | None,
        fallback: dict | None = None,
    ) -> Optional["GaussianCamera"]:
        merged: dict = {}
        if fallback:
            merged.update(fallback)
        if data:
            merged.update(data)
        required = ("width", "height", "fx", "fy", "cx", "cy")
        if not all(key in merged and merged[key] is not None for key in required):
            return None
        return cls(
            width=int(merged["width"]),
            height=int(merged["height"]),
            fx=float(merged["fx"]),
            fy=float(merged["fy"]),
            cx=float(merged["cx"]),
            cy=float(merged["cy"]),
            near_plane=float(merged.get("near_plane", 0.01)),
            far_plane=float(merged.get("far_plane", 1.0e4)),
        )

    def intrinsic(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass
class GaussianRenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray


@dataclass
class GaussianModelTensors:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: Optional[int]


class GsplatGaussianRenderer:
    """Small gsplat renderer for classic 3DGS PLY maps."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._cache: dict[str, GaussianModelTensors] = {}

    def render(
        self,
        ply_path: Path,
        pose_c2w: np.ndarray,
        camera: GaussianCamera,
    ) -> GaussianRenderResult:
        model = self._load_model(Path(ply_path))
        c2w = np.asarray(pose_c2w, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"pose_c2w must be 4x4, got {c2w.shape}")

        w2c = np.linalg.inv(c2w).astype(np.float32)
        viewmats = torch.from_numpy(w2c).to(self.device).reshape(1, 4, 4)
        Ks = torch.from_numpy(camera.intrinsic()).to(self.device).reshape(1, 3, 3)
        with torch.inference_mode():
            renders, alphas, _ = rasterization(
                model.means,
                model.quats,
                model.scales,
                model.opacities,
                model.colors,
                viewmats,
                Ks,
                camera.width,
                camera.height,
                near_plane=camera.near_plane,
                far_plane=camera.far_plane,
                backgrounds=None,
                # PnP unprojection needs expected surface depth rather than
                # alpha-accumulated depth.
                render_mode="RGB+ED",
                sh_degree=model.sh_degree,
                packed=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        render_np = renders[0].detach().float().cpu().numpy()
        alpha_np = alphas[0, ..., 0].detach().float().cpu().numpy()
        rgb = np.clip(render_np[..., :3], 0.0, 1.0)
        depth = render_np[..., 3].astype(np.float32)
        return GaussianRenderResult(
            rgb=(rgb * 255.0).astype(np.uint8),
            depth=depth,
            alpha=alpha_np.astype(np.float32),
        )

    def _load_model(self, ply_path: Path) -> GaussianModelTensors:
        key = str(ply_path.expanduser().resolve(strict=False))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not ply_path.exists():
            raise FileNotFoundError(f"Gaussian ply not found: {ply_path}")
        model = self._read_3dgs_ply(ply_path)
        self._cache[key] = model
        logging.info(
            "Loaded gsplat gaussian map: path=%s points=%d sh_degree=%s",
            ply_path,
            model.means.shape[0],
            model.sh_degree,
        )
        return model

    def _read_3dgs_ply(self, ply_path: Path) -> GaussianModelTensors:
        ply = PlyData.read(str(ply_path))
        vertex = ply["vertex"].data
        names = vertex.dtype.names or ()
        required = {
            "x",
            "y",
            "z",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        }
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"PLY is missing required 3DGS fields: {missing}")

        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        raw_scales = np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]],
            axis=1,
        ).astype(np.float32)
        if np.nanmedian(raw_scales) < 0.0:
            scales = np.exp(raw_scales)
        else:
            scales = raw_scales
        quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)
        quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1.0e-8)
        raw_opacities = np.asarray(vertex["opacity"], dtype=np.float32)
        if np.nanmin(raw_opacities) < 0.0 or np.nanmax(raw_opacities) > 1.0:
            opacities = self._sigmoid(raw_opacities)
        else:
            opacities = np.clip(raw_opacities, 0.0, 1.0)

        f_dc_names = sorted(
            (name for name in names if name.startswith("f_dc_")),
            key=self._field_suffix,
        )
        f_rest_names = sorted(
            (name for name in names if name.startswith("f_rest_")),
            key=self._field_suffix,
        )
        if len(f_dc_names) >= 3:
            dc = np.stack([vertex[name] for name in f_dc_names[:3]], axis=1).astype(np.float32)
            if f_rest_names:
                rest = np.stack([vertex[name] for name in f_rest_names], axis=1).astype(np.float32)
                coeff_count = 1 + rest.shape[1] // 3
                colors = np.zeros((means.shape[0], coeff_count, 3), dtype=np.float32)
                colors[:, 0, :] = dc
                # Standard 3DGS PLY stores all coefficients for R, then G, then B.
                # gsplat rasterization expects the in-memory layout (coefficient, RGB).
                rest_coeff_count = coeff_count - 1
                colors[:, 1:, :] = rest[:, : rest_coeff_count * 3].reshape(
                    means.shape[0],
                    3,
                    rest_coeff_count,
                ).transpose(0, 2, 1)
                sh_degree = int(round(np.sqrt(coeff_count) - 1))
            else:
                colors = np.clip(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
                sh_degree = None
        elif all(name in names for name in ("red", "green", "blue")):
            colors = (
                np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
                / 255.0
            )
            sh_degree = None
        else:
            raise ValueError("PLY must contain either f_dc_* SH fields or red/green/blue colors.")

        return GaussianModelTensors(
            means=torch.from_numpy(means).to(self.device),
            quats=torch.from_numpy(quats).to(self.device),
            scales=torch.from_numpy(scales).to(self.device),
            opacities=torch.from_numpy(opacities).to(self.device),
            colors=torch.from_numpy(colors).to(self.device),
            sh_degree=sh_degree,
        )

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _field_suffix(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[1])
        except Exception:
            return 0
