#!/usr/bin/env python3
"""
Analyze image brightness statistics for one or more VPS databases.

Examples:
  python scripts/analyze_brightness.py /data/map_day/train/rgb
  python scripts/analyze_brightness.py /data/map_day/train/rgb /data/map_night/train/rgb
  python scripts/analyze_brightness.py /data/map_day/train/rgb --recursive
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class BrightnessStats:
    dataset: str
    num_images: int
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    q05: float
    q10: float
    q25: float
    q75: float
    q90: float
    q95: float


@dataclass
class ImageBrightness:
    path: Path
    brightness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze grayscale brightness statistics for one or more image folders."
    )
    parser.add_argument(
        "image_dirs",
        nargs="+",
        help="One or more directories that contain RGB images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional max number of images to analyze from each folder.",
    )
    parser.add_argument(
        "--print-outliers",
        action="store_true",
        help="Print darkest and brightest outlier image names.",
    )
    parser.add_argument(
        "--outlier-zscore",
        type=float,
        default=2.5,
        help="Z-score threshold for brightness outliers.",
    )
    parser.add_argument(
        "--max-outliers",
        type=int,
        default=20,
        help="Maximum number of dark/bright outlier image names to print.",
    )
    return parser.parse_args()


def iter_image_paths(image_dir: Path, recursive: bool) -> Iterable[Path]:
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def compute_image_brightness(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return float(np.mean(image))


def analyze_directory(
    image_dir: Path,
    recursive: bool,
    sample_limit: int | None,
) -> tuple[BrightnessStats, list[ImageBrightness]]:
    image_paths = sorted(iter_image_paths(image_dir, recursive=recursive))
    if sample_limit is not None:
        image_paths = image_paths[:sample_limit]
    if not image_paths:
        raise RuntimeError(f"No supported images found in {image_dir}")

    entries = [ImageBrightness(path=path, brightness=compute_image_brightness(path)) for path in image_paths]
    values = np.array([entry.brightness for entry in entries], dtype=np.float64)
    return BrightnessStats(
        dataset=str(image_dir),
        num_images=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        q05=float(np.quantile(values, 0.05)),
        q10=float(np.quantile(values, 0.10)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
        q90=float(np.quantile(values, 0.90)),
        q95=float(np.quantile(values, 0.95)),
    ), entries


def print_stats(stats: BrightnessStats) -> None:
    print(f"\n=== {stats.dataset} ===")
    print(f"images    : {stats.num_images}")
    print(f"mean      : {stats.mean:.3f}")
    print(f"median    : {stats.median:.3f}")
    print(f"std       : {stats.std:.3f}")
    print(f"min/max   : {stats.min_value:.3f} / {stats.max_value:.3f}")
    print(f"q05/q10   : {stats.q05:.3f} / {stats.q10:.3f}")
    print(f"q25/q75   : {stats.q25:.3f} / {stats.q75:.3f}")
    print(f"q90/q95   : {stats.q90:.3f} / {stats.q95:.3f}")


def print_outliers(
    stats: BrightnessStats,
    entries: list[ImageBrightness],
    zscore_threshold: float,
    max_outliers: int,
) -> None:
    if not entries or stats.std <= 1e-9:
        print("outliers  : unavailable (empty set or zero std)")
        return

    dark_threshold = stats.mean - zscore_threshold * stats.std
    bright_threshold = stats.mean + zscore_threshold * stats.std

    dark_outliers = sorted(
        [entry for entry in entries if entry.brightness <= dark_threshold],
        key=lambda item: item.brightness,
    )
    bright_outliers = sorted(
        [entry for entry in entries if entry.brightness >= bright_threshold],
        key=lambda item: item.brightness,
        reverse=True,
    )

    print("\n--- Outlier Thresholds ---")
    print(f"dark  <= {dark_threshold:.3f}")
    print(f"bright >= {bright_threshold:.3f}")

    print("\n--- Dark Outliers ---")
    if not dark_outliers:
        print("none")
    else:
        for entry in dark_outliers[:max_outliers]:
            print(f"{entry.brightness:8.3f}  {entry.path.name}")

    print("\n--- Bright Outliers ---")
    if not bright_outliers:
        print("none")
    else:
        for entry in bright_outliers[:max_outliers]:
            print(f"{entry.brightness:8.3f}  {entry.path.name}")


def print_two_dataset_hint(left: BrightnessStats, right: BrightnessStats) -> None:
    darker, brighter = sorted([left, right], key=lambda item: item.median)
    threshold_by_median = (darker.median + brighter.median) / 2.0
    threshold_by_iqr_gap = (darker.q75 + brighter.q25) / 2.0
    overlap_low = max(darker.q25, brighter.q25)
    overlap_high = min(darker.q75, brighter.q75)

    print("\n=== Heuristic Threshold Hint ===")
    print(f"darker set : {darker.dataset}")
    print(f"brighter set: {brighter.dataset}")
    print(f"midpoint(median) : {threshold_by_median:.3f}")
    print(f"midpoint(q75/q25): {threshold_by_iqr_gap:.3f}")
    if overlap_low <= overlap_high:
        print(
            f"warning: IQR overlap detected in [{overlap_low:.3f}, {overlap_high:.3f}], "
            "single-threshold switching may be unstable."
        )
    else:
        print("IQR overlap: none")


def main() -> None:
    args = parse_args()
    stats_list: list[BrightnessStats] = []

    for raw_dir in args.image_dirs:
        image_dir = Path(raw_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {image_dir}")
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")
        stats, entries = analyze_directory(
            image_dir=image_dir,
            recursive=args.recursive,
            sample_limit=args.sample_limit,
        )
        stats_list.append(stats)
        print_stats(stats)
        if args.print_outliers:
            print_outliers(
                stats=stats,
                entries=entries,
                zscore_threshold=args.outlier_zscore,
                max_outliers=args.max_outliers,
            )

    if len(stats_list) == 2:
        print_two_dataset_hint(stats_list[0], stats_list[1])


if __name__ == "__main__":
    main()
