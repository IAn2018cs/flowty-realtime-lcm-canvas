import os
import random
from os import path
from contextlib import nullcontext
import time
from sys import platform
import torch

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
is_mac = platform == "darwin"

def should_use_fp16():
    if is_mac:
        return True

    gpu_props = torch.cuda.get_device_properties("cuda")

    if gpu_props.major < 6:
        return False

    nvidia_16_series = ["1660", "1650", "1630"]

    for x in nvidia_16_series:
        if x in gpu_props.name:
            return False

    return True

class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")


def load_models(model_id="stabilityai/sdxl-turbo"):
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image

    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_fp16 = should_use_fp16()

    if use_fp16:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            cache_dir=cache_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        )
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            cache_dir=cache_path,
            safety_checker=None
        )
    device = "mps" if is_mac else "cuda"

    pipe.to(device=device)

    generator = torch.Generator()

    def infer(
            prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=random.randrange(0, 2**63)
    ):
        with torch.inference_mode():
            with torch.autocast("cuda") if device == "cuda" else nullcontext():
                with timer("inference"):
                    return pipe(
                        prompt=prompt,
                        generator=generator.manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]

    return infer