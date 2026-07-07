from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


VGGT_OMEGA_ROOT = Path(__file__).resolve().parent / "third_party" / "vggt-omega"
if str(VGGT_OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_OMEGA_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VGGT / VGGT-Omega inference time.")
    parser.add_argument(
        "--model",
        choices=["vggt", "vggt_omega"],
        default="vggt_omega",
        help="Model family to benchmark.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/panhewei/VPS_online/third_party/vggt/examples/kitchen/images",
        help="Directory containing input images.",
    )
    parser.add_argument("--frame-num", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--omega-resolution", type=int, default=512)
    parser.add_argument("--omega-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--enable-alignment",
        action="store_true",
        help="Enable VGGT-Omega text-alignment head. Use this with the 256 text-alignment checkpoint.",
    )
    parser.add_argument(
        "--simulate-ref-cache",
        action="store_true",
        help="For VGGT-Omega, benchmark preprocessing when ref images are cached and only query is processed live.",
    )
    parser.add_argument(
        "--profile-omega-preprocess",
        action="store_true",
        help="For VGGT-Omega, break CPU preprocessing into open/convert, crop, resize, tensor, and pad/stack timings.",
    )
    return parser.parse_args()


def load_state_dict(checkpoint: str):
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        return state_dict["model"]
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def resolve_images(image_dir: str, frame_num: int) -> list[str]:
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_names = [path for path in image_names if os.path.isfile(path)]
    if not image_names:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_names[:frame_num]


def cuda_event_time(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    for _ in range(runs):
        fn()
    end_time.record()
    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time) / runs


def tensor_size_mib(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def pad_image_tensors_to_common_size(images: list[torch.Tensor]) -> torch.Tensor:
    shapes = {(int(image.shape[-2]), int(image.shape[-1])) for image in images}
    if len(shapes) == 1:
        return torch.stack(images)

    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded_images = []
    for image in images:
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        h_padding = max_height - height
        w_padding = max_width - width
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        if h_padding > 0 or w_padding > 0:
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_images.append(image)
    return torch.stack(padded_images)


def crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Image.Image:
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))

    return image


def round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def omega_target_shape(
    width: int,
    height: int,
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> tuple[int, int]:
    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        token_number = (image_resolution // patch_size) ** 2
        w_patches = np.sqrt(token_number / aspect_ratio)
        h_patches = token_number / w_patches
        w_patches = max(1, int(np.round(w_patches)))
        h_patches = max(1, int(np.round(h_patches)))
        return h_patches * patch_size, w_patches * patch_size

    if aspect_ratio >= 1.0:
        target_h = image_resolution
        target_w = round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        target_w = image_resolution
        target_h = round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return target_h, target_w


def profile_omega_preprocess(
    image_names: list[str],
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> torch.Tensor:
    timings = {
        "open_convert": 0.0,
        "crop": 0.0,
        "resize": 0.0,
        "to_tensor": 0.0,
        "pad_stack": 0.0,
    }
    images: list[torch.Tensor] = []
    to_tensor = TF.ToTensor()

    for image_name in image_names:
        step = time.time()
        with Image.open(image_name) as image:
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert("RGB")
        timings["open_convert"] += time.time() - step

        step = time.time()
        image = crop_to_supported_aspect_ratio(image)
        timings["crop"] += time.time() - step

        step = time.time()
        width, height = image.size
        target_h, target_w = omega_target_shape(
            width=width,
            height=height,
            mode=mode,
            image_resolution=image_resolution,
            patch_size=patch_size,
        )
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        timings["resize"] += time.time() - step

        step = time.time()
        images.append(to_tensor(image))
        timings["to_tensor"] += time.time() - step

    step = time.time()
    stacked = pad_image_tensors_to_common_size(images)
    timings["pad_stack"] += time.time() - step

    total = sum(timings.values())
    print("\n--- omega preprocess profile ---")
    for key in ["open_convert", "crop", "resize", "to_tensor", "pad_stack"]:
        value = timings[key]
        per_image = value / max(len(image_names), 1)
        ratio = (value / total * 100.0) if total > 0 else 0.0
        print(f"{key:12s}: {value:.4f} s  ({per_image:.4f} s/img, {ratio:.1f}%)")
    print(f"profile total: {total:.4f} s")
    print(f"profile shape: {tuple(stacked.shape)}")
    print(f"profile tensor size: {tensor_size_mib(stacked):.2f} MiB")
    return stacked


def benchmark_vggt_omega(args: argparse.Namespace, image_names: list[str], device: torch.device) -> None:
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega(enable_alignment=args.enable_alignment).eval().to(device)
    model.load_state_dict(load_state_dict(args.checkpoint), strict=True)

    preprocess_start = time.time()
    image_names = [str(path) for path in image_names]
    paths_ready_time = time.time()
    images = load_and_preprocess_images(
        image_names,
        mode=args.omega_mode,
        image_resolution=args.omega_resolution,
        patch_size=args.patch_size,
    )
    cpu_preprocess_time = time.time()
    images = images.to(device, non_blocking=True)
    to_device_time = time.time()
    print(f"model       : vggt_omega")
    print(f"alignment   : {args.enable_alignment}")
    print(f"device      : {device}")
    print(f"images      : {len(image_names)}")
    print(f"image_shape : {tuple(images.shape)}")
    print(f"tensor size : {tensor_size_mib(images):.2f} MiB")
    print(f"path prep   : {paths_ready_time - preprocess_start:.4f} s")
    print(f"cpu preprocess: {cpu_preprocess_time - paths_ready_time:.4f} s")
    print(f"to device   : {to_device_time - cpu_preprocess_time:.4f} s")
    print(f"preprocess total: {to_device_time - preprocess_start:.4f} s")

    if args.profile_omega_preprocess:
        profile_omega_preprocess(
            image_names=image_names,
            mode=args.omega_mode,
            image_resolution=args.omega_resolution,
            patch_size=args.patch_size,
        )

    def run_forward():
        with torch.inference_mode():
            predictions = model(images)
            encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
            if args.enable_alignment:
                _ = predictions["text_alignment_embedding"]

    runtime_ms = cuda_event_time(run_forward, warmup=args.warmup, runs=args.runs)
    print(f"forward avg : {runtime_ms:.2f} ms ({runtime_ms / 1000:.4f} s), runs={args.runs}")

    if args.simulate_ref_cache:
        benchmark_vggt_omega_ref_cache(args, image_names, device, model)


def benchmark_vggt_omega_ref_cache(
    args: argparse.Namespace,
    image_names: list[str],
    device: torch.device,
    model: torch.nn.Module,
) -> None:
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if len(image_names) < 2:
        print("\nref-cache simulation skipped: need at least 2 images")
        return

    query_name = image_names[0]
    ref_names = image_names[1:]

    cache_build_start = time.time()
    ref_cache = {}
    for ref_name in ref_names:
        ref_tensor = load_and_preprocess_images(
            [ref_name],
            mode=args.omega_mode,
            image_resolution=args.omega_resolution,
            patch_size=args.patch_size,
        )[0]
        ref_cache[ref_name] = ref_tensor
    cache_build_time = time.time() - cache_build_start

    live_start = time.time()
    query_tensor = load_and_preprocess_images(
        [query_name],
        mode=args.omega_mode,
        image_resolution=args.omega_resolution,
        patch_size=args.patch_size,
    )[0]
    query_preprocess_time = time.time()
    cached_images = [query_tensor, *[ref_cache[name] for name in ref_names]]
    images = pad_image_tensors_to_common_size(cached_images)
    stack_pad_time = time.time()
    images = images.to(device, non_blocking=True)
    to_device_time = time.time()

    print("\n--- ref-cache preprocess simulation ---")
    print(f"query image       : {query_name}")
    print(f"cached refs       : {len(ref_names)}")
    print(f"cache build       : {cache_build_time:.4f} s")
    print(f"query preprocess  : {query_preprocess_time - live_start:.4f} s")
    print(f"stack/pad         : {stack_pad_time - query_preprocess_time:.4f} s")
    print(f"to device         : {to_device_time - stack_pad_time:.4f} s")
    print(f"cached total      : {to_device_time - live_start:.4f} s")
    print(f"cached image_shape: {tuple(images.shape)}")
    print(f"cached tensor size: {tensor_size_mib(images):.2f} MiB")

    def run_forward():
        with torch.inference_mode():
            predictions = model(images)
            encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
            if args.enable_alignment:
                _ = predictions["text_alignment_embedding"]

    runtime_ms = cuda_event_time(run_forward, warmup=args.warmup, runs=args.runs)
    print(f"cached forward avg: {runtime_ms:.2f} ms ({runtime_ms / 1000:.4f} s), runs={args.runs}")


def benchmark_vggt(args: argparse.Namespace, image_names: list[str], device: torch.device) -> None:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT(enable_track=True).eval().to(device)
    model.load_state_dict(load_state_dict(args.checkpoint), strict=False)

    preprocess_start = time.time()
    image_names = [str(path) for path in image_names]
    paths_ready_time = time.time()
    images = load_and_preprocess_images(image_names)
    cpu_preprocess_time = time.time()
    images = images.to(device, non_blocking=True)
    to_device_time = time.time()
    images = images[:, :, :336, :518].clone()
    images = images[None]
    reshape_time = time.time()
    print(f"model       : vggt")
    print(f"device      : {device}")
    print(f"dtype       : {dtype}")
    print(f"images      : {len(image_names)}")
    print(f"image_shape : {tuple(images.shape)}")
    print(f"tensor size : {tensor_size_mib(images):.2f} MiB")
    print(f"path prep   : {paths_ready_time - preprocess_start:.4f} s")
    print(f"cpu preprocess: {cpu_preprocess_time - paths_ready_time:.4f} s")
    print(f"to device   : {to_device_time - cpu_preprocess_time:.4f} s")
    print(f"reshape     : {reshape_time - to_device_time:.4f} s")
    print(f"preprocess total: {reshape_time - preprocess_start:.4f} s")

    def run_aggregator():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                model.aggregator(images)

    runtime_ms = cuda_event_time(run_aggregator, warmup=args.warmup, runs=args.runs)
    print(f"aggregator avg: {runtime_ms:.2f} ms ({runtime_ms / 1000:.4f} s), runs={args.runs}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")
    device = torch.device(args.device)
    image_names = resolve_images(args.image_dir, args.frame_num)

    if args.model == "vggt_omega":
        benchmark_vggt_omega(args, image_names, device)
    else:
        benchmark_vggt(args, image_names, device)


if __name__ == "__main__":
    main()
