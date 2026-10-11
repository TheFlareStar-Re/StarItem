"""训练集复现检查：模型能不能画出合成数据里的指定物品。

For each concept the model draws a few sprites from the caption used in training ("中文名, english name",
synthetic style). CLIP then ranks the concept among all concepts in synth/concepts, the same score used to
filter the training sprites, so the three numbers are comparable: training sprites, this model, other models.

Usage:
  .venv/Scripts/python.exe -X utf8 train/eval_recall.py --run sic12c --step 80000 --names synth/concepts/pilot150.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "synth"))
from build_dataset import direct_ranks  # noqa: E402
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from generate import load_concepts  # noqa: E402
from model import UNet  # noqa: E402
from pilot import Clip  # noqa: E402
from train import TextEncoder, autocast, to_rgba_uint8  # noqa: E402


def summary(label, ranks):
    r = np.array(ranks)
    print(f"{label:28} n={len(r):5}  top20 {np.mean(r <= 20):4.0%}  top100 {np.mean(r <= 100):4.0%}  "
          f"top300 {np.mean(r <= 300):4.0%}  median rank {int(np.median(r))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--names", type=Path, required=True, help="text file, one English concept name per line")
    ap.add_argument("--n", type=int, default=4, help="images per concept")
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--source", type=float, default=1.0)
    ap.add_argument("--train", type=Path, nargs="*", default=[], help="datasets whose sprites of these concepts are scored too")
    ap.add_argument("--buckets", type=Path, nargs="*", default=[],
                    help="datasets used to count training sprites per concept; results are also reported by that count")
    ap.add_argument("--sheet", type=Path, help="save the first 40 concepts as an image")
    ap.add_argument("--per-concept", type=Path, help="write each concept's ranks to this CSV")
    args = ap.parse_args()

    concepts = load_concepts()
    index = {c["en"]: i for i, c in enumerate(concepts)}
    names = [n for n in args.names.read_text(encoding="utf-8").splitlines() if n in index]
    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda")

    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    ctx, mask = text.encode([f"{concepts[index[n]]['zh']}, {n}".lower() for n in names])
    null_ctx, null_mask = text.encode([""])
    del text
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(run / f"ema_{args.step:06d}.pt", map_location=dev, weights_only=True))
    ac = cosine_alphas_cumprod().to(dev)
    sprites = []
    g = torch.Generator(device=dev).manual_seed(0)
    for i in range(0, len(names), 64):
        c, m = ctx[i:i + 64].repeat_interleave(args.n, 0), mask[i:i + 64].repeat_interleave(args.n, 0)
        with autocast(dev):
            x = ddim_sample(model, c, m, null_ctx, null_mask, 16, 0, ac, steps=args.steps, guidance=args.guidance,
                            generator=g, source=float(args.source))
        sprites += list(to_rgba_uint8(x))
    del model
    torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    clip = Clip(dev, ["x"])
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    all_names = [c["en"] for c in concepts]
    for folder in args.train:
        rows = [json.loads(line) for line in (folder / "metadata.jsonl").open(encoding="utf-8")]
        rows = [r for r in rows if r["name_en"] in set(names)]
        if rows:
            imgs = [np.asarray(Image.open(folder / r["file"]).convert("RGBA")) for r in rows]
            summary(f"training sprites, {folder.name}", direct_ranks(clip, tokenizer, imgs, all_names, [index[r["name_en"]] for r in rows]))
    ranks = direct_ranks(clip, tokenizer, sprites, all_names, [index[n] for n in names for _ in range(args.n)])
    summary(f"{args.run} @ {args.step}", ranks)
    best = np.array(ranks).reshape(len(names), args.n).min(axis=1)
    print(f"concepts with at least one image in the top 100: {np.mean(best <= 100):.0%}; none in the top 300: "
          f"{np.mean(best > 300):.0%}")

    if args.per_concept:
        import csv

        per = np.array(ranks).reshape(len(names), args.n)
        with args.per_concept.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name_en", "name_zh", "best_rank", "median_rank"])
            for n, row in zip(names, per):
                w.writerow([n, concepts[index[n]]["zh"], int(row.min()), int(np.median(row))])

    if args.buckets:
        # 按每个物品的训练图数量分组，看"图越多是否画得越像"
        count = dict.fromkeys(names, 0)
        for folder in args.buckets:
            for line in (folder / "metadata.jsonl").open(encoding="utf-8"):
                name = json.loads(line)["name_en"]
                if name in count:
                    count[name] += 1
        per = np.array(ranks).reshape(len(names), args.n)
        for lo, hi in ((0, 3), (4, 9), (10, 15), (16, 99)):
            pick = [i for i, n in enumerate(names) if lo <= count[n] <= hi]
            if pick:
                summary(f"  {lo}-{hi} training sprites ({len(pick)} concepts)", per[pick].ravel())

    if args.sheet:
        show, cell = min(40, len(names)), 68
        sheet = Image.new("RGBA", (2 * (args.n * cell + 16), (show + 1) // 2 * cell + 4), (58, 58, 68, 255))
        for i in range(show * args.n):
            row, k = divmod(i, args.n)
            tile = Image.fromarray(sprites[i], "RGBA").resize((64, 64), Image.NEAREST)
            sheet.alpha_composite(tile, ((row % 2) * (args.n * cell + 16) + 4 + k * cell, 4 + (row // 2) * cell))
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.sheet)
        print(f"sheet (two concepts per row, in list order): {args.sheet}")


if __name__ == "__main__":
    main()
