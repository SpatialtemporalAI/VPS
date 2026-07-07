#!/usr/bin/env python3
"""Select clean VPS keyframes from dense SLAM-style rgb/poses folders."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class FrameRecord:
    image_path: Path
    pose_path: Path
    pose: np.ndarray
    brightness: float
    blur: float
    gray_std: float
    black_ratio: float
    white_ratio: float
    quality_ok: bool
    quality_reason: str
    keep: bool = False
    reject_reason: str = ""
    translation_from_last: Optional[float] = None
    rotation_from_last_deg: Optional[float] = None
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dense VPS rgb/poses into a cleaner keyframe database."
    )
    parser.add_argument("--rgb-dir", required=True, help="Input rgb directory.")
    parser.add_argument("--poses-dir", required=True, help="Input poses directory with 4x4 c2w txt files.")
    parser.add_argument("--out-dir", required=True, help="Output directory containing rgb/, poses/, selection_report.csv.")
    parser.add_argument("--min-translation", type=float, default=0.4, help="Minimum translation from last kept frame.")
    parser.add_argument("--min-rotation-deg", type=float, default=15.0, help="Minimum rotation angle from last kept frame.")
    parser.add_argument("--grid-size", type=float, default=0.75, help="XY grid size for density limiting.")
    parser.add_argument("--max-per-grid", type=int, default=10, help="Maximum kept frames per XY grid cell.")
    parser.add_argument("--brightness-min", type=float, default=40.0)
    parser.add_argument("--brightness-max", type=float, default=220.0)
    parser.add_argument("--blur-min", type=float, default=50.0, help="Minimum Laplacian variance.")
    parser.add_argument("--gray-std-min", type=float, default=8.0, help="Minimum grayscale std to reject low-texture frames.")
    parser.add_argument("--black-ratio-max", type=float, default=0.5)
    parser.add_argument("--white-ratio-max", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Only write report; do not copy files.")
    return parser.parse_args()


def image_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**18, path.name


def find_image_pose_pairs(rgb_dir: Path, poses_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted(
        [path for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=image_sort_key,
    )
    pairs = []
    for image_path in image_paths:
        pose_path = poses_dir / f"{image_path.stem}.txt"
        if pose_path.exists():
            pairs.append((image_path, pose_path))
    return pairs


def load_pose(path: Path) -> np.ndarray:
    import numpy as np

    pose = np.loadtxt(path).reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"Invalid pose shape for {path}: {pose.shape}")
    return pose


def rotation_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    import numpy as np

    rel = pose_a[:3, :3].T @ pose_b[:3, :3]
    cos_angle = (np.trace(rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def compute_quality(image_path: Path, args: argparse.Namespace) -> tuple[float, float, float, float, float, bool, str]:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0, 0.0, 1.0, 0.0, False, "image_read_failed"

    brightness = float(np.mean(image))
    gray_std = float(np.std(image))
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    black_ratio = float(np.mean(image <= 10))
    white_ratio = float(np.mean(image >= 245))

    reasons = []
    if brightness < args.brightness_min:
        reasons.append("too_dark")
    if brightness > args.brightness_max:
        reasons.append("too_bright")
    if blur < args.blur_min:
        reasons.append("too_blurry")
    if gray_std < args.gray_std_min:
        reasons.append("low_texture")
    if black_ratio > args.black_ratio_max:
        reasons.append("too_much_black")
    if white_ratio > args.white_ratio_max:
        reasons.append("too_much_white")

    return brightness, blur, gray_std, black_ratio, white_ratio, not reasons, "|".join(reasons)


def build_records(args: argparse.Namespace) -> list[FrameRecord]:
    rgb_dir = Path(args.rgb_dir)
    poses_dir = Path(args.poses_dir)
    pairs = find_image_pose_pairs(rgb_dir, poses_dir)
    if not pairs:
        raise RuntimeError(f"No image/pose pairs found in {rgb_dir} and {poses_dir}")

    records = []
    for image_path, pose_path in pairs:
        pose = load_pose(pose_path)
        brightness, blur, gray_std, black_ratio, white_ratio, quality_ok, quality_reason = compute_quality(
            image_path,
            args,
        )
        records.append(
            FrameRecord(
                image_path=image_path,
                pose_path=pose_path,
                pose=pose,
                brightness=brightness,
                blur=blur,
                gray_std=gray_std,
                black_ratio=black_ratio,
                white_ratio=white_ratio,
                quality_ok=quality_ok,
                quality_reason=quality_reason,
            )
        )
    return records


def select_keyframes(records: list[FrameRecord], args: argparse.Namespace) -> list[FrameRecord]:
    import numpy as np

    kept: list[FrameRecord] = []
    grid_counts: dict[tuple[int, int], int] = {}
    last_kept_pose: Optional[np.ndarray] = None

    for record in records:
        x = float(record.pose[0, 3])
        y = float(record.pose[1, 3])
        grid_key = (
            int(math.floor(x / args.grid_size)),
            int(math.floor(y / args.grid_size)),
        )
        record.grid_x, record.grid_y = grid_key

        if not record.quality_ok:
            record.reject_reason = record.quality_reason
            continue

        if grid_counts.get(grid_key, 0) >= args.max_per_grid:
            record.reject_reason = "grid_full"
            continue

        if last_kept_pose is None:
            record.keep = True
            record.reject_reason = ""
        else:
            translation = float(np.linalg.norm(record.pose[:3, 3] - last_kept_pose[:3, 3]))
            rotation_deg = rotation_angle_deg(last_kept_pose, record.pose)
            record.translation_from_last = translation
            record.rotation_from_last_deg = rotation_deg

            if translation >= args.min_translation or rotation_deg >= args.min_rotation_deg:
                record.keep = True
            else:
                record.reject_reason = "too_close"

        if record.keep:
            kept.append(record)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
            last_kept_pose = record.pose

    return kept


def copy_outputs(records: list[FrameRecord], out_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    out_rgb = out_dir / "rgb"
    out_poses = out_dir / "poses"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_poses.mkdir(parents=True, exist_ok=True)

    for record in records:
        if not record.keep:
            continue
        shutil.copy2(record.image_path, out_rgb / record.image_path.name)
        shutil.copy2(record.pose_path, out_poses / record.pose_path.name)


def write_report(records: list[FrameRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "selection_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "pose",
                "keep",
                "reject_reason",
                "brightness",
                "blur",
                "gray_std",
                "black_ratio",
                "white_ratio",
                "translation_from_last",
                "rotation_from_last_deg",
                "grid_x",
                "grid_y",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image": record.image_path.name,
                    "pose": record.pose_path.name,
                    "keep": int(record.keep),
                    "reject_reason": record.reject_reason,
                    "brightness": f"{record.brightness:.6f}",
                    "blur": f"{record.blur:.6f}",
                    "gray_std": f"{record.gray_std:.6f}",
                    "black_ratio": f"{record.black_ratio:.6f}",
                    "white_ratio": f"{record.white_ratio:.6f}",
                    "translation_from_last": "" if record.translation_from_last is None else f"{record.translation_from_last:.6f}",
                    "rotation_from_last_deg": "" if record.rotation_from_last_deg is None else f"{record.rotation_from_last_deg:.6f}",
                    "grid_x": record.grid_x,
                    "grid_y": record.grid_y,
                }
            )


def print_summary(records: list[FrameRecord], kept: list[FrameRecord], out_dir: Path, dry_run: bool) -> None:
    reasons: dict[str, int] = {}
    for record in records:
        if record.keep:
            continue
        reasons[record.reject_reason or "unknown"] = reasons.get(record.reject_reason or "unknown", 0) + 1

    print(f"input frames : {len(records)}")
    print(f"kept frames  : {len(kept)}")
    print(f"reject frames: {len(records) - len(kept)}")
    print(f"out dir      : {out_dir}")
    print(f"dry run      : {dry_run}")
    if reasons:
        print("reject reasons:")
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason}: {count}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    records = build_records(args)
    kept = select_keyframes(records, args)
    write_report(records, out_dir)
    copy_outputs(records, out_dir, dry_run=args.dry_run)
    print_summary(records, kept, out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
