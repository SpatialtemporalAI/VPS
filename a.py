#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one localization request to the VPS service.")
    parser.add_argument("--url", default="http://127.0.0.1:6001", help="Service base url")
    parser.add_argument("--image", required=True, help="Path to query RGB image")
    parser.add_argument("--depth", default=None, help="Optional depth file path (.png or .npy)")
    parser.add_argument("--robot-id", default="0", help="Robot id used by the service")
    parser.add_argument("--map-id", default=None, help="Optional target map id")
    parser.add_argument(
        "--save-map",
        default="a_result_map.png",
        help="Output path for decoded response map; use empty string to skip saving",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of requests to send sequentially")
    parser.add_argument("--interval", type=float, default=0.0, help="Sleep seconds between repeated requests")
    parser.add_argument(
        "--robot-ids",
        default=None,
        help="Comma-separated robot ids for alternating requests, e.g. 0,1,2",
    )
    parser.add_argument(
        "--map-ids",
        default=None,
        help="Optional comma-separated map ids aligned with --robot-ids; use empty item to skip a robot map binding",
    )
    parser.add_argument(
        "--save-map-each",
        action="store_true",
        help="When repeating, save each returned map with an index suffix",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent workers. 1 means sequential execution.",
    )
    return parser


def send_localize_request(
    base_url: str,
    image_path: Path,
    robot_id: str,
    depth_path: Path | None = None,
    map_id: str | None = None,
    save_map_path: str | None = None,
) -> int:
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2
    if depth_path is not None and not depth_path.exists():
        print(f"depth not found: {depth_path}", file=sys.stderr)
        return 2

    data = {"robot_id": robot_id}
    if map_id:
        data["map_id"] = map_id

    files = {}
    handles = []
    try:
        image_handle = open(image_path, "rb")
        handles.append(image_handle)
        files["image"] = (image_path.name, image_handle)

        if depth_path is not None:
            depth_handle = open(depth_path, "rb")
            handles.append(depth_handle)
            files["depth"] = (depth_path.name, depth_handle)

        start = time.time()
        response = requests.post(f"{base_url.rstrip('/')}/localize", files=files, data=data, timeout=60)
        elapsed = time.time() - start
    finally:
        for handle in handles:
            handle.close()

    print(f"status: {response.status_code}")
    print(f"elapsed: {elapsed:.3f}s")

    try:
        payload = response.json()
    except json.JSONDecodeError:
        print(response.text)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if response.status_code != 200:
        return 1

    pose = payload.get("pose")
    if pose:
        print(
            f"pose: x={pose.get('x'):.6f}, y={pose.get('y'):.6f}, theta={pose.get('theta'):.6f}"
        )

    map_b64 = payload.get("map_png_b64")
    if map_b64 and save_map_path:
        image_bytes = base64.b64decode(map_b64)
        map_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if map_image is None:
            print("failed to decode returned map", file=sys.stderr)
            return 1
        out_path = Path(save_map_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), map_image)
        print(f"saved response map: {out_path}")

    return 0


def build_save_map_path(base_path: str | None, index: int, repeat: int, save_each: bool) -> str | None:
    if not base_path:
        return base_path
    if repeat <= 1 or not save_each:
        return base_path
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}_{index:03d}{path.suffix}"))


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",")]


def build_robot_schedule(
    repeat: int,
    default_robot_id: str,
    default_map_id: str | None,
    robot_ids_raw: str | None,
    map_ids_raw: str | None,
) -> list[tuple[str, str | None]]:
    robot_ids = parse_csv_list(robot_ids_raw)
    if not robot_ids:
        robot_ids = [default_robot_id]

    map_ids = parse_csv_list(map_ids_raw)
    if map_ids and len(map_ids) not in {1, len(robot_ids)}:
        raise ValueError("--map-ids must have length 1 or match --robot-ids")
    if len(map_ids) == 1 and len(robot_ids) > 1:
        map_ids = map_ids * len(robot_ids)
    if not map_ids:
        map_ids = [default_map_id] * len(robot_ids)

    schedule: list[tuple[str, str | None]] = []
    for index in range(repeat):
        slot = index % len(robot_ids)
        map_id = map_ids[slot] if slot < len(map_ids) else default_map_id
        schedule.append((robot_ids[slot], map_id or None))
    return schedule


def main() -> int:
    args = build_parser().parse_args()
    repeat = max(1, args.repeat)
    schedule = build_robot_schedule(
        repeat=repeat,
        default_robot_id=str(args.robot_id),
        default_map_id=args.map_id,
        robot_ids_raw=args.robot_ids,
        map_ids_raw=args.map_ids,
    )
    exit_code = 0
    success_count = 0
    elapsed_list: list[float] = []
    overall_start = time.time()

    concurrent = max(1, args.concurrent)

    def run_one(index: int, robot_id: str, map_id: str | None) -> tuple[int, float]:
        print(f"\n=== request {index + 1}/{repeat} robot={robot_id} map={map_id} ===")
        start = time.time()
        code = send_localize_request(
            base_url=args.url,
            image_path=Path(args.image),
            depth_path=Path(args.depth) if args.depth else None,
            robot_id=robot_id,
            map_id=map_id,
            save_map_path=build_save_map_path(args.save_map, index, repeat, args.save_map_each),
        )
        return code, time.time() - start

    if concurrent == 1:
        for index, (robot_id, map_id) in enumerate(schedule):
            code, elapsed = run_one(index, robot_id, map_id)
            elapsed_list.append(elapsed)
            if code == 0:
                success_count += 1
            else:
                exit_code = code

            if index < repeat - 1 and args.interval > 0:
                time.sleep(args.interval)
    else:
        with ThreadPoolExecutor(max_workers=concurrent, thread_name_prefix="a-py") as executor:
            future_to_index = {}
            for index, (robot_id, map_id) in enumerate(schedule):
                future = executor.submit(run_one, index, robot_id, map_id)
                future_to_index[future] = index
                if args.interval > 0 and index < repeat - 1:
                    time.sleep(args.interval)

            for future in as_completed(future_to_index):
                code, elapsed = future.result()
                elapsed_list.append(elapsed)
                if code == 0:
                    success_count += 1
                else:
                    exit_code = code

    total_elapsed = time.time() - overall_start
    print("\n=== summary ===")
    print(f"requests: {repeat}")
    print(f"concurrent: {concurrent}")
    print(f"success: {success_count}")
    print(f"failed: {repeat - success_count}")
    print(f"total_elapsed: {total_elapsed:.3f}s")
    if elapsed_list:
        print(f"avg_elapsed: {sum(elapsed_list) / len(elapsed_list):.3f}s")
        print(f"min_elapsed: {min(elapsed_list):.3f}s")
        print(f"max_elapsed: {max(elapsed_list):.3f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
