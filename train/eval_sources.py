"""比较"来源/画风"条件的不同取值：0 = 手绘模组贴图，1 = 合成图，中间值是两者的混合。

Writes runs/<run>/eval/sources_<step>.png: one block per setting, one row per prompt.

Usage:
  python -X utf8 train/eval_sources.py --run sic11 --step 120000 --mix 0,0.5,1
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
from train import TextEncoder, autocast, to_rgba_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# 前 4 个是模组贴图里常见的物品，后面是只在合成数据里出现过的
PROMPTS = ["diamond sword", "铜锭", "red potion", "铁镐", "roast turkey", "感恩节火鸡", "拿铁", "acoustic guitar",
           "红色的自行车", "smartphone", "blue umbrella", "汉堡", "寿司", "茶壶", "rubber duck", "金鱼"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sic11")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--mix", default="0,0.5,1", help="weights of the synthetic style to compare")
    ap.add_argument("--prompts", help="prompts separated by |; default is the built-in list")
    ap.add_argument("--out", help="output file name inside runs/<run>/eval")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    prompts = [p.strip() for p in args.prompts.split("|")] if args.prompts else PROMPTS
    ctx, mask = text.encode([p.lower() for p in prompts])
    ctx, mask = ctx.repeat_interleave(args.n, 0), mask.repeat_interleave(args.n, 0)
    null_ctx, null_mask = text.encode([""])
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(run / f"ema_{args.step:06d}.pt", map_location=dev, weights_only=True))
    base = model.src_emb.weight.detach().clone()
    ac = cosine_alphas_cumprod().to(dev)

    mixes = [float(m) for m in args.mix.split(",")]
    S, pad = 5, 3
    cell = 16 * S + pad
    block = args.n * cell + 14
    sheet = Image.new("RGBA", (len(mixes) * block + 6, len(prompts) * cell + 6), (58, 58, 68, 255))
    for bi, w in enumerate(mixes):
        # 把 0 号来源的嵌入换成两种来源的加权平均，再按 0 号来源采样
        with torch.no_grad():
            model.src_emb.weight[0] = (1 - w) * base[0] + w * base[1]
        g = torch.Generator(device=dev).manual_seed(0)
        with autocast(dev):
            x = ddim_sample(model, ctx, mask, null_ctx, null_mask, 16, 0, ac, steps=args.steps, guidance=args.guidance,
                            generator=g, source=0)
        for i, im in enumerate(to_rgba_uint8(x)):
            row, k = divmod(i, args.n)
            tile = Image.fromarray(im, "RGBA").resize((16 * S, 16 * S), Image.NEAREST)
            sheet.alpha_composite(tile, (6 + bi * block + k * cell, 6 + row * cell))
    out = run / "eval" / (args.out or f"sources_{args.step:06d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"blocks left to right: synthetic weight {mixes}; rows: {prompts}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
