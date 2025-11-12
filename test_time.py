import torch
import glob
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from torch.nn.attention import sdpa_kernel, SDPBackend

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(device)
print(dtype)
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
model_path = "/ssd1/phw/checkpoints/model.pt"

model.load_state_dict(torch.load(model_path))
model.eval()  # Ensure model is in evaluation mode
model.to(device)
# Load and preprocess example images (replace with your own image paths)
frame_num = 10
image_names = glob.glob(os.path.join("/ssd1/phw/xichuang/ref/rgb/", "*"))

image_names = image_names[:frame_num]
images = load_and_preprocess_images(image_names).to(device, non_blocking=True)
images = images[:, :, :336, :518].clone()    # match the setting of Table


with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        images = images[None]  # add batch dimension
        
        ########################## 
        # Multiple warm-up iterations for better GPU optimization
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, 
        #                             enable_mem_efficient=False,
        #                             enable_math=False):
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(3):
            _, _ = model.aggregator(images)
        torch.cuda.synchronize()  # Ensure warm-up is complete
        #########################
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Benchmark with multiple runs for more accurate timing
        # for rough estimate, just run once
        num_runs = 10
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, 
        #                             enable_mem_efficient=False,
        #                             enable_math=True):
        start_time.record()
        for _ in range(num_runs):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        end_time.record()
        torch.cuda.synchronize()
        
        runtime_ms = start_time.elapsed_time(end_time) / num_runs  # Average time per run
        runtime_sec = runtime_ms / 1000  # Convert ms to seconds
        print(f"Time taken (avg of {num_runs} runs): {runtime_ms:.2f} ms ({runtime_sec:.4f} s)")