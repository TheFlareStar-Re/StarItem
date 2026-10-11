"""在 SDXL 上训练一个 LoRA，让它直接画出 16x16 网格的模组物品贴图。

Training images are the hand-drawn mod sprites (dataset/clean), each pixel enlarged to a 32x32 block on a
magenta background (512x512). After training, SDXL draws any object in that style, and the 16x16 sprite is
read back block by block instead of shrinking detailed art.

Uses only the models already in the Hugging Face cache; nothing is downloaded.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/train_lora.py --steps 3000
  .venv/Scripts/python.exe -X utf8 synth/train_lora.py --steps 9000 --resume
"""
import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import cached_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "lora16"
RES, GRID = 512, 16
KEY = (255, 0, 255)  # 背景用品红：贴图里几乎不会出现，生成后按这个颜色抠掉
TRIGGER = "mc16 item, "


def render(rgba):
    """16x16 RGBA -> 512x512 RGB：每个像素放大成 32x32 的色块，透明处填品红。"""
    a = rgba[..., 3:] > 0
    rgb = np.where(a, rgba[..., :3], np.array(KEY, dtype=np.uint8))
    return rgb.repeat(RES // GRID, axis=0).repeat(RES // GRID, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    device = torch.device("cuda")
    torch.manual_seed(0)
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    base = cached_dir("stabilityai/stable-diffusion-xl-base-1.0", "model_index.json")
    vae = AutoencoderKL.from_pretrained(cached_dir("madebyollin/sdxl-vae-fp16-fix", "config.json"), torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to(device)

    rows = [json.loads(line) for line in (ROOT / "dataset" / "clean" / "metadata.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["size"] == 16]
    names = [r["name_en"].lower() for r in rows]
    print(f"{len(rows)} sprites, {len(set(names))} distinct names", flush=True)

    # 1. 图像先过 VAE 编码成潜变量，训练时不再需要 VAE
    t0 = time.time()
    cache = OUT / "cache.pt"
    cached = torch.load(cache, weights_only=True) if cache.exists() else None
    if cached is not None and cached["names"] != names:
        cached = None  # 数据集变了，缓存作废
    latents = []
    with torch.no_grad():
        for i in range(0, 0 if cached else len(rows), 8):
            batch = np.stack([render(np.asarray(Image.open(ROOT / "dataset" / "clean" / r["file"]).convert("RGBA")))
                              for r in rows[i:i + 8]])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, torch.float16) / 127.5 - 1.0
            latents.append((pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor).half())
    latents = cached["latents"].to(device) if cached else torch.cat(latents)
    print(f"encoded images in {time.time() - t0:.0f}s, latents {tuple(latents.shape)}", flush=True)

    # 2. 提示词过两个文本编码器，结果放在内存里（很大，留在 CPU 上）
    t0 = time.time()
    unique = sorted(set(names)) + [""]
    slot = {n: i for i, n in enumerate(unique)}
    embeds, pooled = [], []
    with torch.no_grad():
        for i in range(0, 0 if cached else len(unique), 64):
            texts = [TRIGGER + n if n else "" for n in unique[i:i + 64]]
            e, _, p, _ = pipe.encode_prompt(prompt=texts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
            embeds.append(e.half().cpu())
            pooled.append(p.half().cpu())
    if cached:
        embeds, pooled = cached["embeds"], cached["pooled"]
    else:
        embeds, pooled = torch.cat(embeds), torch.cat(pooled)
        # 这两步要十几分钟，结果存下来，续训和重跑时直接读
        torch.save({"names": names, "latents": latents.cpu(), "embeds": embeds, "pooled": pooled}, cache)
    name_idx = torch.tensor([slot[n] for n in names])
    print(f"encoded {len(unique)} prompts in {time.time() - t0:.0f}s", flush=True)

    unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # 文本编码器和 VAE 用完就放掉，显存留给训练
    del pipe, vae
    gc.collect()
    torch.cuda.empty_cache()
    unet.enable_gradient_checkpointing()
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    # 3. 给 UNet 的注意力层加 LoRA，只训练这部分
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    step = 0
    if args.resume:
        unet_state = {k.replace("unet.", ""): v for k, v in load_file(OUT / "pytorch_lora_weights.safetensors").items()}
        from diffusers.utils import convert_unet_state_dict_to_peft
        set_peft_model_state_dict(unet, convert_unet_state_dict_to_peft(unet_state), adapter_name="default")
        step = json.loads((OUT / "state.json").read_text())["step"]
        print(f"resumed from step {step}", flush=True)
    cast_training_params(unet, dtype=torch.float32)
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    time_ids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=device, dtype=torch.float16)

    def save(s):
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionXLPipeline.save_lora_weights(OUT, unet_lora_layers=state)
        (OUT / "state.json").write_text(json.dumps({"step": s, "trigger": TRIGGER, "res": RES, "grid": GRID, "key": KEY}))

    unet.train()
    running, t_last, s_last = None, time.time(), step
    n = len(rows)
    while step < args.steps:
        idx = torch.randint(0, n, (args.batch,))
        x0 = latents[idx.to(device)].float()
        cap = name_idx[idx]
        # 一部分样本不给提示词，生成时才能用 classifier-free guidance
        cap = torch.where(torch.rand(args.batch) < 0.1, torch.full_like(cap, slot[""]), cap)
        noise = torch.randn_like(x0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch,), device=device)
        noisy = scheduler.add_noise(x0, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, encoder_hidden_states=embeds[cap].to(device),
                        added_cond_kwargs={"text_embeds": pooled[cap].to(device), "time_ids": time_ids.expand(args.batch, -1)}).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if not math.isfinite(running):
            sys.exit(f"loss became non-finite at step {step}")
        if step % 100 == 0:
            now = time.time()
            speed = (step - s_last) / (now - t_last)
            t_last, s_last = now, step
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:5}/{args.steps} loss {running:.4f} {speed:.2f} it/s eta {(args.steps - step) / speed / 60:.0f} min, peak {peak:.1f} GB", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(step)
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
