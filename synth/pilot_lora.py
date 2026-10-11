"""试跑：用 LoRA 直接生成 16x16 网格的贴图，看 50 个物品的效果。

Writes synth/pilot_out/lora_sheet.png: for each concept, the raw 512px outputs (left) and the 16x16 sprites
read back from them (right).

Usage:
  .venv/Scripts/python.exe -X utf8 synth/pilot_lora.py
  .venv/Scripts/python.exe -X utf8 synth/pilot_lora.py --fast --limit 20
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_sprites import check, load_pipeline, read_grid  # noqa: E402
from pilot import CONCEPTS, Clip, on_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "synth" / "pilot_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(CONCEPTS))
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="8 steps with the LCM LoRA instead of 25 normal steps")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=0)
    ap.add_argument("--out", default="lora_sheet.png")
    args = ap.parse_args()
    steps = args.steps or (8 if args.fast else 25)
    guidance = args.guidance or (1.0 if args.fast else 6.0)

    device = torch.device("cuda")
    pipe, state = load_pipeline(device, args.fast)
    concepts = CONCEPTS[:args.limit]
    generator = torch.Generator(device=device).manual_seed(0)
    raw, sprites, spreads, answers, seconds = [], [], [], [], []
    for ci, (thing, zh) in enumerate(concepts):
        name = re.sub(r"^(a|an) ", "", thing)
        t0 = time.time()
        images = pipe(prompt=[state["trigger"] + name] * args.per, num_inference_steps=steps, guidance_scale=guidance,
                      height=state["res"], width=state["res"], generator=generator).images
        if ci > 0:
            seconds.append((time.time() - t0) / args.per)
        for image in images:
            sprite, spread = read_grid(image, state)
            raw.append(image.resize((128, 128), Image.LANCZOS))
            sprites.append(None if check(sprite) else sprite)
            spreads.append(spread)
            answers.append(ci)
        print(f"[{ci + 1:2}/{len(concepts)}] {zh} {name}: " + " ".join(check(read_grid(i, state)[0]) or "ok" for i in images), flush=True)
    del pipe
    torch.cuda.empty_cache()

    usable = [i for i, s in enumerate(sprites) if s is not None]
    clip = Clip(device, [f"a pixel art icon of {thing}" for thing, _ in concepts])
    ranks = clip.rank([on_gray(sprites[i], 224) for i in usable], [answers[i] for i in usable]) if usable else []

    cell, pad = 128, 6
    sheet = Image.new("RGBA", (args.per * 2 * (cell + pad) + pad + 14, len(concepts) * (cell + pad) + pad), (58, 58, 68, 255))
    for i, (r, s) in enumerate(zip(raw, sprites)):
        row, k = divmod(i, args.per)
        y = pad + row * (cell + pad)
        sheet.paste(r, (pad + k * (cell + pad), y))
        if s is not None:
            sheet.paste(Image.fromarray(on_gray(s, cell)), (14 + pad + (args.per + k) * (cell + pad), y))
    sheet.save(OUT / args.out)
    n = len(sprites)
    print(f"\nimages: {n}, usable: {len(usable)} ({len(usable) / n:.0%}), mean within-cell spread: {np.mean(spreads):.1f}")
    if ranks:
        print(f"CLIP on 16px sprites, out of {len(concepts)} concepts: top-1 {np.mean(np.array(ranks) <= 1):.0%}, "
              f"top-5 {np.mean(np.array(ranks) <= 5):.0%}")
    if seconds:
        print(f"speed: {np.mean(seconds):.2f} s per image")
    print(f"sheet: {OUT / args.out}")


if __name__ == "__main__":
    main()
