from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


DEFAULT_IMGPOSE_PATH = Path("/data/nvme0n1/phw/烯创26-0624/ImgPose.txt")
DEFAULT_IMAGE_ROOT = Path("/data/nvme0n1/phw/烯创26-0624/undistort")
DEFAULT_OUTPUT_DIR = Path("/data/nvme0n1/phw/烯创26-0624/day/train")


@dataclass
class ImagePoseRecord:
    image_rel_path: Path
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Xichuang ImgPose.txt data to VPS train/rgb + train/poses format."
        )
    )
    parser.add_argument(
        "--imgpose",
        type=Path,
        default=DEFAULT_IMGPOSE_PATH,
        help=f"Path to ImgPose.txt. Default: {DEFAULT_IMGPOSE_PATH}",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Root containing left/right image folders. Default: {DEFAULT_IMAGE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output VPS train directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--camera",
        choices=("all", "left", "right"),
        default="all",
        help="Which camera records to export from ImgPose.txt. Default: all",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting output index. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing rgb/pose files.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing source images instead of failing.",
    )
    return parser.parse_args()


def load_imgpose_records(imgpose_path: Path, camera: str) -> list[ImagePoseRecord]:
    records: list[ImagePoseRecord] = []
    with imgpose_path.open("r", encoding="utf-8", errors="replace") as file:
        header = file.readline().strip().split()
        expected = [
            "index",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "qx",
            "qy",
            "qz",
            "qw",
            "timestamp",
        ]
        if header[: len(expected)] != expected:
            raise ValueError(
                f"Unexpected ImgPose header in {imgpose_path}: {' '.join(header)}"
            )

        for line_no, line in enumerate(file, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                raise ValueError(f"Invalid ImgPose line {line_no}: {line}")

            image_rel_path = Path(parts[0])
            camera_name = image_rel_path.parts[0] if image_rel_path.parts else ""
            if camera != "all" and camera_name != camera:
                continue

            translation = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            )
            quaternion = np.array(
                [float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10])],
                dtype=np.float64,
            )
            timestamp = float(parts[11])
            records.append(
                ImagePoseRecord(
                    image_rel_path=image_rel_path,
                    translation=translation,
                    quaternion_xyzw=quaternion,
                    timestamp=timestamp,
                )
            )
    return records


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError(f"Invalid zero quaternion: {quaternion}")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def record_to_pose(record: ImagePoseRecord) -> np.ndarray:
    """Build VPS camera-to-world pose from ImgPose row.

    ImgPose quaternions already match the VPS rotation convention produced by
    xichuang2vps_trans.py after its original R @ diag(1, -1, -1) correction.
    Do not apply that correction again here.
    """

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_xyzw_to_rotation(record.quaternion_xyzw)
    pose[:3, 3] = record.translation
    return pose


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace existing files."
        )


def export_records(
    records: Iterable[ImagePoseRecord],
    image_root: Path,
    output_dir: Path,
    start_index: int,
    overwrite: bool,
    skip_missing: bool,
) -> tuple[int, int]:
    rgb_dir = output_dir / "rgb"
    poses_dir = output_dir / "poses"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for output_idx, record in enumerate(records, start=start_index):
        source_image = image_root / record.image_rel_path
        if not source_image.exists():
            if skip_missing:
                print(f"Warning: missing image, skipped: {source_image}")
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing image: {source_image}")

        rgb_path = rgb_dir / f"{output_idx}.png"
        pose_path = poses_dir / f"{output_idx}.txt"
        ensure_output_path(rgb_path, overwrite=overwrite)
        ensure_output_path(pose_path, overwrite=overwrite)

        pose = record_to_pose(record)
        np.savetxt(pose_path, pose, fmt="%.8f")
        shutil.copyfile(source_image, rgb_path)
        exported += 1
        print(
            f"[{exported}] {record.image_rel_path} -> rgb/{output_idx}.png, "
            f"poses/{output_idx}.txt"
        )
    return exported, skipped


def main() -> None:
    args = parse_args()
    records = load_imgpose_records(args.imgpose, camera=args.camera)
    if not records:
        raise RuntimeError(f"No records found in {args.imgpose} for camera={args.camera}")

    exported, skipped = export_records(
        records=records,
        image_root=args.image_root,
        output_dir=args.output_dir,
        start_index=args.start_index,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
    )
    print(
        f"\n处理完成: exported={exported}, skipped={skipped}, "
        f"output={args.output_dir}/rgb 和 {args.output_dir}/poses"
    )


if __name__ == "__main__":
    main()
