from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

from vggt_omega.utils.load_fn import load_and_preprocess_images


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class VGGTOmegaPreprocessConfig:
    mode: str = "max_size"
    image_resolution: int = 256
    patch_size: int = 16


def collect_image_paths(image_dir: Path, recursive: bool = False) -> list[Path]:
    image_dir = Path(image_dir)
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported tensor dtype: {dtype_name}")


def preprocess_vggt_omega_images(
    image_paths: Sequence[Path],
    config: VGGTOmegaPreprocessConfig,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Preprocess images with the same rules used by VGGT-Omega.

    The upstream preprocessing returns normalized float32 tensors in [0, 1].
    ``storage_dtype`` only controls how the cached tensor is stored.
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    tensors = load_and_preprocess_images(
        [Path(path) for path in image_paths],
        mode=config.mode,
        image_resolution=config.image_resolution,
        patch_size=config.patch_size,
    )
    return tensors.to(dtype=storage_dtype)


def build_vggt_omega_ref_cache(
    image_paths: Sequence[Path],
    config: VGGTOmegaPreprocessConfig,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> dict:
    image_paths = [Path(path) for path in image_paths]
    tensors = preprocess_vggt_omega_images(
        image_paths,
        config,
        storage_dtype=storage_dtype,
    )
    return {
        "format": "vggt_omega_ref_tensor_cache_v1",
        "config": asdict(config),
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "names": [path.name for path in image_paths],
        "paths": [str(path) for path in image_paths],
        "mtimes_ns": [path.stat().st_mtime_ns for path in image_paths],
        "tensors": tensors.cpu(),
    }


def save_vggt_omega_ref_cache(cache: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)


def load_vggt_omega_ref_cache(cache_path: Path, *, map_location: str | torch.device = "cpu") -> dict:
    cache = torch.load(Path(cache_path), map_location=map_location)
    if cache.get("format") != "vggt_omega_ref_tensor_cache_v1":
        raise ValueError(f"Unsupported VGGT-Omega ref cache format: {cache.get('format')}")
    return cache


class VGGTOmegaRefTensorCache:
    """CPU tensor index for fixed VGGT-Omega reference images."""

    def __init__(self, caches: Sequence[dict]):
        self._by_path: dict[str, torch.Tensor] = {}
        self._by_name: dict[str, torch.Tensor | None] = {}
        self.configs: list[dict] = []
        self.storage_dtypes: list[str] = []
        self.num_images = 0

        for cache in caches:
            self.configs.append(dict(cache.get("config", {})))
            self.storage_dtypes.append(str(cache.get("storage_dtype", "")))
            tensors = cache["tensors"].cpu()
            names = cache["names"]
            paths = cache["paths"]
            for idx, raw_path in enumerate(paths):
                tensor = tensors[idx]
                self._by_path[self._normalize_path(raw_path)] = tensor
                name = names[idx]
                if name in self._by_name:
                    self._by_name[name] = None
                else:
                    self._by_name[name] = tensor
            self.num_images += len(names)

    @classmethod
    def from_paths(cls, cache_paths: Sequence[Path]) -> "VGGTOmegaRefTensorCache":
        return cls([load_vggt_omega_ref_cache(path, map_location="cpu") for path in cache_paths])

    def validate_config(self, expected: VGGTOmegaPreprocessConfig) -> None:
        expected_dict = asdict(expected)
        mismatches = [config for config in self.configs if config != expected_dict]
        if mismatches:
            raise ValueError(
                "VGGT-Omega ref cache preprocess config does not match current model config. "
                f"expected={expected_dict}, found={mismatches[0]}"
            )

    def get_many(self, ref_paths: Iterable[Path]) -> tuple[list[torch.Tensor], list[str]]:
        tensors: list[torch.Tensor] = []
        missing: list[str] = []
        for path in ref_paths:
            tensor = self.get(path)
            if tensor is None:
                missing.append(str(path))
            else:
                tensors.append(tensor)
        return tensors, missing

    def get(self, ref_path: Path) -> torch.Tensor | None:
        path = Path(ref_path)
        tensor = self._by_path.get(self._normalize_path(path))
        if tensor is not None:
            return tensor
        # Basename fallback is only safe when that basename is unique across all caches.
        return self._by_name.get(path.name)

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve(strict=False))


def get_ref_tensors_from_cache(cache: dict, ref_names: Iterable[str]) -> torch.Tensor:
    name_to_index = {name: idx for idx, name in enumerate(cache["names"])}
    indices = []
    for name in ref_names:
        if name not in name_to_index:
            raise KeyError(f"Ref image '{name}' is missing from VGGT-Omega cache.")
        indices.append(name_to_index[name])
    return cache["tensors"][indices]


def pad_image_tensors_to_common_size(
    tensors: Sequence[torch.Tensor],
    *,
    pad_value: float = 1.0,
) -> torch.Tensor:
    """Pad ``C,H,W`` image tensors to a shared size and stack as ``N,C,H,W``."""
    if not tensors:
        raise ValueError("tensors cannot be empty")

    max_h = max(int(tensor.shape[-2]) for tensor in tensors)
    max_w = max(int(tensor.shape[-1]) for tensor in tensors)
    first = tensors[0]
    prefix_shape = tuple(first.shape[:-2])
    same_shape = True
    for tensor in tensors:
        if tuple(tensor.shape[:-2]) != prefix_shape:
            raise ValueError(
                f"All tensors must share leading shape {prefix_shape}, got {tuple(tensor.shape[:-2])}"
            )
        if int(tensor.shape[-2]) != max_h or int(tensor.shape[-1]) != max_w:
            same_shape = False

    if same_shape:
        return torch.stack(tensors)

    output_shape = (len(tensors), *prefix_shape, max_h, max_w)
    output = torch.full(
        output_shape,
        pad_value,
        dtype=first.dtype,
        device=first.device,
    )
    for idx, tensor in enumerate(tensors):
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        top = (max_h - height) // 2
        left = (max_w - width) // 2
        output[idx, ..., top : top + height, left : left + width] = tensor
    return output


def pad_image_tensors_to_common_size_legacy(
    tensors: Sequence[torch.Tensor],
    *,
    pad_value: float = 1.0,
) -> torch.Tensor:
    """Reference implementation kept for debugging equivalence."""
    if not tensors:
        raise ValueError("tensors cannot be empty")

    max_h = max(int(tensor.shape[-2]) for tensor in tensors)
    max_w = max(int(tensor.shape[-1]) for tensor in tensors)
    padded = []
    for tensor in tensors:
        h_padding = max_h - int(tensor.shape[-2])
        w_padding = max_w - int(tensor.shape[-1])
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=pad_value,
            )
        padded.append(tensor)
    return torch.stack(padded)
