"""用训练好的 LoRA 生成贴图，并把 512x512 的输出逐格读回成 16x16。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import cached_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LORA = ROOT / "runs" / "lora16"


def load_pipeline(device, fast=False):
    """SDXL + 我们训练的 LoRA。fast=True 时再叠加 LCM LoRA，用 8 步出图。只读本地缓存。"""
    from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(cached_dir("madebyollin/sdxl-vae-fp16-fix", "config.json"), torch_dtype=torch.float16)
    base = cached_dir("stabilityai/stable-diffusion-xl-base-1.0", "model_index.json")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.load_lora_weights(LORA, weight_name="pytorch_lora_weights.safetensors", adapter_name="mc16")
    if fast:
        lcm = "pytorch_lora_weights.safetensors"
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(cached_dir("latent-consistency/lcm-lora-sdxl", lcm), weight_name=lcm, adapter_name="lcm")
        pipe.set_adapters(["mc16", "lcm"], adapter_weights=[1.0, 1.0])
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    return pipe.to(device), json.loads((LORA / "state.json").read_text())


def read_grid(image, state, tolerance=48):
    """512x512 的 PIL 图 -> (16x16 RGBA 贴图, 网格整齐度)。

    每格取中间一半区域的中位色，避开色块边缘的过渡。背景色不假定一定是训练时的品红
    （生成图的背景有时偏粉、偏紫甚至发白），而是取最外一圈格子里最常见的颜色，
    再从边缘向内把与它相近、且连通的格子去掉。
    整齐度是格内颜色的平均偏差，越小说明生成图越贴合 16x16 网格。
    """
    a = np.asarray(image.convert("RGB")).astype(np.int16)
    grid = state["grid"]
    block = a.shape[0] // grid
    q = block // 4
    cells = a.reshape(grid, block, grid, block, 3)[:, q:block - q, :, q:block - q].transpose(0, 2, 1, 3, 4).reshape(grid, grid, -1, 3)
    color = np.median(cells, axis=2)
    spread = float(np.abs(cells - color[:, :, None]).mean())

    edge = np.zeros((grid, grid), dtype=bool)
    edge[0] = edge[-1] = edge[:, 0] = edge[:, -1] = True
    border = color[edge]
    # 最外圈里和最多格子颜色相近的那个颜色就是背景色
    near = np.abs(border[:, None] - border[None]).max(axis=-1) < tolerance
    votes = near.sum(axis=1)
    background = border[near[votes.argmax()]].mean(axis=0)
    if votes.max() < 0.35 * len(border):
        background = np.array(state["key"], dtype=np.float32)  # 边缘很乱时退回训练用的背景色
    dist = np.abs(color - background).max(axis=-1)
    close = dist < tolerance
    transparent = np.zeros((grid, grid), dtype=bool)
    stack = [(y, x) for y, x in zip(*np.nonzero(edge & close))]
    for y, x in stack:
        transparent[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < grid and 0 <= nx < grid and close[ny, nx] and not transparent[ny, nx]:
                transparent[ny, nx] = True
                stack.append((ny, nx))
    # 物体内部和背景色几乎一样的格子（甜甜圈的孔）也是透明的
    transparent |= dist < tolerance / 2
    out = np.zeros((grid, grid, 4), dtype=np.uint8)
    out[..., :3] = np.where(transparent[..., None], 0, color).astype(np.uint8)
    out[..., 3] = np.where(transparent, 0, 255)
    return out, spread


def check(sprite):
    """返回这张贴图的问题说明，没有问题返回 None。"""
    opaque = sprite[..., 3] > 0
    n = int(opaque.sum())
    if n < 20:
        return "empty"
    if n > 250:
        return "no_background"
    # 物体里还留着成片的背景品红（常见于屏幕之类的平面），这种图不要
    magenta = np.abs(sprite[..., :3].astype(np.int16) - np.array([255, 0, 255])).max(axis=-1) < 70
    if int((magenta & opaque).sum()) > 3:
        return "key_color_inside"
    return None
