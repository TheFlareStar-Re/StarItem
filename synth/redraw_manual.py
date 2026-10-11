"""试验：让我们自己的 LoRA 以外部模型的图为底稿重画，得到模组画风的 16x16 贴图，并且一张底稿出多张变体。

For each source image in dataset/manual_raw16/records.jsonl, compares the direct 16x16 conversion with img2img
redraws at several strengths. Writes dataset/manual_raw16/redraw_test.png and prints CLIP ranks.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/redraw_manual.py --per-item 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import direct_ranks  # noqa: E402
from generate import load_concepts  # noqa: E402
from lora_sprites import check, load_pipeline, read_grid  # noqa: E402
from pilot import Clip  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset" / "manual_raw16"


def as_canvas(sprite, state):
    """16x16 RGBA -> 训练时那种 512x512 的图：每个像素一个色块，透明处填背景色。"""
    rgb = np.where(sprite[..., 3:] > 0, sprite[..., :3], np.array(state["key"], dtype=np.uint8))
    block = state["res"] // state["grid"]
    return Image.fromarray(rgb.repeat(block, axis=0).repeat(block, axis=1).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", type=int, default=1, help="source images per item to try")
    ap.add_argument("--strengths", default="0.4,0.55,0.7")
    ap.add_argument("--n", type=int, default=4, help="redraws per source image and strength")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=6.0)
    args = ap.parse_args()

    from diffusers import StableDiffusionXLImg2ImgPipeline

    device = torch.device("cuda")
    pipe, state = load_pipeline(device)
    pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components).to(device)
    pipe.set_progress_bar_config(disable=True)
    strengths = [float(s) for s in args.strengths.split(",")]

    records = [json.loads(line) for line in (SRC / "records.jsonl").open(encoding="utf-8")]
    records = [r for r in records if r["file"] and r["k"] < args.per_item]
    rows = []  # (record, direct sprite, {strength: [sprites]})
    for r in records:
        direct = np.asarray(Image.open(SRC / "sprite" / r["file"]).convert("RGBA"))
        init = as_canvas(direct, state)
        redraws = {}
        for s in strengths:
            g = torch.Generator(device=device).manual_seed(100 + r["id"])
            images = pipe(prompt=[state["trigger"] + r["en"]] * args.n, image=[init] * args.n, strength=s,
                          num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images
            redraws[s] = [None if check(sp) else sp for sp in (read_grid(im, state)[0] for im in images)]
        rows.append((r, direct, redraws))
        print(r["en"], {s: sum(x is not None for x in v) for s, v in redraws.items()}, flush=True)
    del pipe
    torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    concepts = load_concepts()
    index = {c["en"]: i for i, c in enumerate(concepts)}
    clip = Clip(device, ["x"])
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    names = [c["en"] for c in concepts]

    def ranks(pairs):
        return np.array(direct_ranks(clip, tokenizer, [s for s, _ in pairs], names, [index[n] for _, n in pairs]))

    d = ranks([(direct, r["en"]) for r, direct, _ in rows])
    print(f"\ndirect conversion: median rank {int(np.median(d))}, top100 {np.mean(d <= 100):.0%}, top300 {np.mean(d <= 300):.0%}")
    for s in strengths:
        pairs = [(sp, r["en"]) for r, _, redraws in rows for sp in redraws[s] if sp is not None]
        total = len(rows) * args.n
        k = ranks(pairs)
        print(f"redraw strength {s}: usable {len(pairs)}/{total}, median rank {int(np.median(k))}, "
              f"top100 {np.mean(k <= 100):.0%}, top300 {np.mean(k <= 300):.0%}")

    cell = 72
    sheet = Image.new("RGB", (cell + 10 + len(strengths) * (args.n * cell + 10), len(rows) * (cell + 4)), (58, 58, 68))
    for y, (r, direct, redraws) in enumerate(rows):
        im = Image.fromarray(direct, "RGBA").resize((cell - 4, cell - 4), Image.NEAREST)
        sheet.paste(im, (2, y * (cell + 4) + 2), im)
        for si, s in enumerate(strengths):
            for k, sp in enumerate(redraws[s]):
                if sp is not None:
                    im = Image.fromarray(sp, "RGBA").resize((cell - 4, cell - 4), Image.NEAREST)
                    sheet.paste(im, (cell + 10 + si * (args.n * cell + 10) + k * cell + 2, y * (cell + 4) + 2), im)
    sheet.save(SRC / "redraw_test.png")
    print(f"sheet: {SRC / 'redraw_test.png'} (first column: direct conversion; then strengths {strengths})")


if __name__ == "__main__":
    main()
