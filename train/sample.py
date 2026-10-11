"""用训练好的模型按提示词生成贴图；可选地把每张结果和训练集里最像的一张并排输出，用来检查是否在背图。

Usage:
  python -X utf8 train/sample.py "ruby sword" "glowing blue potion"
  python -X utf8 train/sample.py --nn "diamond sword" --guidance 2 --n 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from model import UNet  # noqa: E402
from train import SIZE_INDEX, TextEncoder, autocast, save_grid, to_rgba_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_train_images(data, size):
    rows = [json.loads(line) for line in (data / "metadata.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["size"] == size]
    imgs = np.stack([np.asarray(Image.open(data / r["file"]).convert("RGBA")) for r in rows])
    return imgs, rows


def main():
    ap = argparse.ArgumentParser(description="Generate item textures from prompts.")
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--run", default="base16")
    ap.add_argument("--ckpt", help="EMA weights file; default is the latest ema_*.pt in the run")
    ap.add_argument("--n", type=int, default=3, help="images per prompt")
    ap.add_argument("--size", type=int, default=16, choices=sorted(SIZE_INDEX))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=int, default=0, help="style: 0 = hand-drawn mod sprites, 1 = synthetic")
    ap.add_argument("--nn", action="store_true", help="also show the nearest training image for each result")
    ap.add_argument("--out", type=Path, help="output grid PNG; default runs/<run>/gen/<seed>.png")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else sorted(run.glob("ema_*.pt"))[-1]
    model = UNet(**cfg["model"]).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], device)
    prompts = [p.lower() for p in args.prompts]  # 训练时名称统一转了小写
    ctx, mask = text.encode(prompts)
    null_ctx, null_mask = text.encode([""])
    ctx, mask = ctx.repeat_interleave(args.n, 0), mask.repeat_interleave(args.n, 0)
    g = torch.Generator(device=device).manual_seed(args.seed)
    with autocast(device):
        x = ddim_sample(model, ctx, mask, null_ctx, null_mask, args.size, SIZE_INDEX[args.size],
                        cosine_alphas_cumprod().to(device), steps=args.steps, guidance=args.guidance, generator=g,
                        source=args.source)
    out = to_rgba_uint8(x)

    cols = args.n
    if args.nn:
        train, rows = load_train_images(ROOT / "dataset" / "clean", args.size)
        # 透明像素不比较颜色：RGB 先乘上 alpha
        flat = lambda a: (a[..., :3].astype(np.float32) * (a[..., 3:] > 0)).reshape(len(a), -1) / 255.0  # noqa: E731
        tf, of = torch.from_numpy(flat(train)).to(device), torch.from_numpy(flat(out)).to(device)
        dist, idx = torch.cdist(of, tf).min(dim=1)
        rms = (dist / (of.shape[1] ** 0.5)).cpu().numpy()  # 每通道的均方根差，0 表示完全相同
        idx = idx.cpu().numpy()
        for i, p in enumerate(prompts):
            sl = slice(i * args.n, (i + 1) * args.n)
            names = [f"{rows[j]['name_en']} [{rows[j]['slug']}] {d:.3f}" for j, d in zip(idx[sl], rms[sl])]
            print(f"{p}: " + " | ".join(names))
        # 每张生成图右边紧跟它最近的训练图
        out = np.stack([im for i in range(len(out)) for im in (out[i], train[idx[i]])])
        cols = args.n * 2
        print(f"rms to nearest training image: median {np.median(rms):.3f}, min {rms.min():.3f}, "
              f"share below 0.05: {(rms < 0.05).mean():.0%}")
    path = args.out or run / "gen" / f"seed{args.seed}.png"
    save_grid(out, cols, path, scale=6 if args.size == 16 else 3)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
