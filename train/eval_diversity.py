"""评估一个 run 的各个检查点：同一提示词多张图之间的差异，以及和训练图的接近程度。

Usage:
  python -X utf8 train/eval_diversity.py --run bi16
  python -X utf8 train/eval_diversity.py --run bi16 --steps 10000,20000,30000 --grid 30000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from model import UNet  # noqa: E402
from sample import load_train_images  # noqa: E402
from train import TextEncoder, autocast, save_grid, to_rgba_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# 都是训练集里没有的组合
PROMPTS_EN = ["purple sword", "strawberry cake", "electric staff", "frozen fish", "golden key", "green potion",
              "bone dagger", "crystal amulet", "burnt toast", "silver hammer", "magic scroll", "fire gem",
              "wooden shield", "magic wand", "ice axe", "pumpkin pie"]
PROMPTS_ZH = ["紫色的剑", "草莓蛋糕", "闪电法杖", "冰冻的鱼", "金钥匙", "绿色药水", "骨头匕首", "水晶护身符",
              "烤焦的面包", "银锤", "魔法卷轴", "火焰宝石", "木盾", "魔杖", "冰斧", "南瓜派"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bi16")
    ap.add_argument("--steps", default="10000,20000,30000")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--source", type=int, default=0, help="style: 0 = hand-drawn mod sprites, 1 = synthetic")
    ap.add_argument("--grid", type=int, help="also save an image grid for this checkpoint step")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    sets = {"en": PROMPTS_EN} if cfg["lang"] == "en" else {"en": PROMPTS_EN, "zh": PROMPTS_ZH}
    null_ctx, null_mask = text.encode([""])
    train, _ = load_train_images(ROOT / "dataset" / "clean", 16)
    flat = lambda a: (a[..., :3].astype(np.float32) * (a[..., 3:] > 0)).reshape(len(a), -1) / 255.0  # noqa: E731
    tf = torch.from_numpy(flat(train)).to(dev)
    ac = cosine_alphas_cumprod().to(dev)

    print("step   lang | diversity | rms to nearest train | near-copies(<0.05)")
    for step in [int(s) for s in args.steps.split(",")]:
        model = UNet(**cfg["model"]).to(dev).eval()
        model.load_state_dict(torch.load(run / f"ema_{step:06d}.pt", map_location=dev, weights_only=True))
        for lang, prompts in sets.items():
            ctx, mask = text.encode(prompts)
            ctx, mask = ctx.repeat_interleave(args.n, 0), mask.repeat_interleave(args.n, 0)
            g = torch.Generator(device=dev).manual_seed(0)
            with autocast(dev):
                x = ddim_sample(model, ctx, mask, null_ctx, null_mask, 16, 0, ac, guidance=args.guidance, generator=g,
                                source=args.source)
            out = to_rgba_uint8(x)
            of = torch.from_numpy(flat(out)).to(dev)
            scale = of.shape[1] ** 0.5
            nn = torch.cdist(of, tf).min(1).values / scale
            o = of.view(len(prompts), args.n, -1)
            div = ((torch.cdist(o, o) / scale).sum((1, 2)) / (args.n * (args.n - 1))).mean().item()
            print(f"{step:6d} {lang:4} | {div:.3f} | {nn.median().item():.3f} | {(nn < 0.05).float().mean().item():.0%}")
            if args.grid == step:
                save_grid(out, args.n, run / "eval" / f"novel_{lang}_{step:06d}.png")


if __name__ == "__main__":
    main()
