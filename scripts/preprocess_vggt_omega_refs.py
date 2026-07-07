#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vps.utils.vggt_omega_preprocess import (
    VGGTOmegaPreprocessConfig,
    build_vggt_omega_ref_cache,
    collect_image_paths,
    dtype_from_name,
    save_vggt_omega_ref_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess fixed reference images into a VGGT-Omega tensor cache."
    )
    parser.add_argument("image_dir", type=Path, help="Reference rgb directory.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt cache path.",
    )
    parser.add_argument(
        "--mode",
        choices=["max_size", "balanced"],
        default="max_size",
        help="VGGT-Omega preprocess mode.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=256,
        help="VGGT-Omega image_resolution.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="VGGT-Omega patch size.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Tensor dtype used inside the saved cache. float32 is safest.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.image_dir, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    config = VGGTOmegaPreprocessConfig(
        mode=args.mode,
        image_resolution=args.image_resolution,
        patch_size=args.patch_size,
    )
    storage_dtype = dtype_from_name(args.storage_dtype)

    start = time.time()
    cache = build_vggt_omega_ref_cache(
        image_paths,
        config,
        storage_dtype=storage_dtype,
    )
    build_time = time.time() - start
    save_vggt_omega_ref_cache(cache, args.output)

    tensors = cache["tensors"]
    size_mib = tensors.numel() * tensors.element_size() / 1024 / 1024
    print(f"images        : {len(image_paths)}")
    print(f"tensor shape  : {tuple(tensors.shape)}")
    print(f"storage dtype : {cache['storage_dtype']}")
    print(f"tensor size   : {size_mib:.2f} MiB")
    print(f"build time    : {build_time:.3f} s")
    print(f"output        : {args.output}")


if __name__ == "__main__":
    main()
