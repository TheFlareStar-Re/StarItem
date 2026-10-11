"""手绘（模组）风格这一侧的评测，和 eval_bind.py 互补。

Two prompt sets, both drawn in the hand-drawn style:
  seen   names of hand-drawn training sprites. Score: where the sprite(s) with that name rank among all
         hand-drawn sprites by CLIP image similarity to the generated image (1 = it is the nearest).
  novel  material x item-type combinations that are not in the training names ("闪电法杖"). Scores:
         sprite-likeness (similarity to the nearest hand-drawn sprite; broken images score low) and
         type match (does the nearest hand-drawn sprite's name contain the item-type word), and
         overfilled: the share of images that cover more of the tile than 95% of the hand-drawn sprites of that
         item type do. Broken images (the item plus a stray block of color) show up here; a faithful model scores 5%.

Usage:
  .venv/Scripts/python.exe -X utf8 train/eval_mod.py --runs sic16@150000 sic17@150000 --sheet export/dev/mod_sheet.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "synth"))
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from eval_bind import embed  # noqa: E402
from model import UNet  # noqa: E402
from pilot import Clip  # noqa: E402
from train import TextEncoder, TokenMap, autocast, to_rgba_uint8  # noqa: E402

MATERIALS = ["钻石", "红宝石", "铜", "钢", "黑曜石", "翡翠", "紫水晶", "火焰", "冰霜", "闪电", "雷电", "暗影", "彩虹", "龙骨",
             "水晶", "星星", "月亮", "太阳", "海洋", "熔岩", "骨头", "木", "金", "银", "铁", "蓝宝石", "末影", "青铜", "樱花", "南瓜"]
TYPES = ["剑", "镐", "斧", "锄", "铲", "弓", "法杖", "魔杖", "匕首", "盾牌", "头盔", "胸甲", "靴子", "护符", "戒指", "项链", "锭",
         "齿轮", "药水", "锤", "钥匙", "粉", "板", "棒", "箭", "皇冠", "书"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run@step, the first is the reference")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--count", type=int, default=400, help="prompts per set")
    ap.add_argument("--sheet", type=Path, help="save the first novel prompts of every run side by side")
    args = ap.parse_args()
    dev = torch.device("cuda")

    base = ROOT / "dataset" / "clean"
    rows = [json.loads(line) for line in (base / "metadata.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["size"] == 16 and r["name_zh"]]
    clip = Clip(dev, ["x"])
    feats = embed(clip, [np.asarray(Image.open(base / r["file"]).convert("RGBA")) for r in rows])
    zh = [r["name_zh"].lower() for r in rows]
    by_name = {}
    for i, n in enumerate(zh):
        by_name.setdefault(n, []).append(i)

    rng = random.Random(0)
    seen = rng.sample(sorted(by_name), args.count)
    # 这个类型词在手绘集里至少有 10 张图，"最近的图是不是同一类"才有意义
    types = [t for t in TYPES if sum(t in n for n in zh) >= 10]
    combos = [(m, t) for m in MATERIALS for t in types if m + t not in by_name]
    novel = rng.sample(combos, min(args.count, len(combos)))
    type_mask = {t: torch.tensor([t in n for n in zh], device=dev) for t in types}
    fill_of = np.array([(np.asarray(Image.open(base / r["file"]).convert("RGBA"))[..., 3] > 0).mean() for r in rows])
    fill_limit = {t: np.percentile(fill_of[type_mask[t].cpu().numpy()], 95) for t in types}
    print(f"hand-drawn sprites {len(rows)}; seen prompts {len(seen)}; novel prompts {len(novel)} over {len(types)} item types", flush=True)

    ac = cosine_alphas_cumprod().to(dev)
    tiles = []
    for spec in args.runs:
        run, step = spec.split("@")
        folder = ROOT / "runs" / run
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
        prompts = seen + [m + t for m, t in novel]
        ctx, mask = text.encode(prompts)
        null_ctx, null_mask = text.encode([""])
        tok = null_tok = None
        if cfg["model"].get("num_tokens"):
            tm = TokenMap.load(folder / "tokens.json")
            tok, null_tok = tm.encode(text.tok, prompts, cfg["max_len"], dev), tm.encode(text.tok, [""], cfg["max_len"], dev)
        del text
        model = UNet(**cfg["model"]).to(dev).eval()
        model.load_state_dict(torch.load(folder / f"ema_{int(step):06d}.pt", map_location=dev, weights_only=True))
        g = torch.Generator(device=dev).manual_seed(0)
        sprites = []
        for i in range(0, len(prompts), 64):
            rep = lambda x: x[i:i + 64].repeat_interleave(args.n, 0)
            with autocast(dev):
                x = ddim_sample(model, rep(ctx), rep(mask), null_ctx, null_mask, 16, 0, ac, steps=30, guidance=5.5, generator=g,
                                source=0.0, tok=None if tok is None else rep(tok), null_tok=null_tok)
            sprites += list(to_rgba_uint8(x))
        del model
        torch.cuda.empty_cache()
        sims = embed(clip, sprites) @ feats.T  # (提示词数 * n, 手绘图数)
        k = len(seen) * args.n

        ranks = []
        for j, name in enumerate(seen):
            own = torch.tensor(by_name[name], device=dev)
            block = sims[j * args.n:(j + 1) * args.n]
            best = block[:, own].max(1).values
            ranks += ((block > best[:, None]).sum(1) + 1).tolist()
        ranks = np.array(ranks)
        like = sims[k:].max(1).values.cpu().numpy()
        nearest = sims[k:].argmax(1)
        match = np.array([bool(type_mask[t][nearest[j * args.n + q]]) for j, (_, t) in enumerate(novel) for q in range(args.n)])
        # 同类里最像的那张有多像：比"最近的一张恰好同类"宽松，受同类图多少的影响小
        own_type = np.array([float(sims[k + j * args.n + q][type_mask[t]].max()) for j, (_, t) in enumerate(novel) for q in range(args.n)])
        fill = np.array([(sp[..., 3] > 0).mean() for sp in sprites[k:]])
        over = np.array([fill[j * args.n + q] > fill_limit[t] for j, (_, t) in enumerate(novel) for q in range(args.n)])
        print(f"{spec:16} seen: median rank {int(np.median(ranks))}, top1 {np.mean(ranks <= 1):.0%}, top10 {np.mean(ranks <= 10):.0%} | "
              f"novel: sprite-likeness median {np.median(like):.3f}, 5th pct {np.percentile(like, 5):.3f}, below 0.80: {np.mean(like < 0.80):.1%}; "
              f"type match {match.mean():.0%}; similarity to best same-type sprite {np.median(own_type):.3f}; overfilled {over.mean():.1%}", flush=True)
        pick = {t: [j * args.n + q for j, (_, tt) in enumerate(novel) if tt == t for q in range(args.n)] for t in types}
        print("   by type (match / overfilled): " + "  ".join(f"{t} {match[i].mean():.0%}/{over[i].mean():.0%}" for t, i in pick.items()), flush=True)
        tiles.append(sprites[k:])

    if args.sheet:
        from PIL import ImageDraw, ImageFont
        show, cell, pad = min(40, len(novel)), 40, 2
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
        block = args.n * (cell + pad) + 10
        colw = 84 + len(args.runs) * block + 8
        sheet = Image.new("RGB", (2 * colw, (show + 1) // 2 * (cell + pad) + 20), (58, 58, 68))
        d = ImageDraw.Draw(sheet)
        for c in range(2):
            for r_i, spec in enumerate(args.runs):
                d.text((c * colw + 84 + r_i * block, 2), spec.split("@")[0], fill=(255, 220, 120), font=font)
        for j in range(show):
            x0, y = (j % 2) * colw, 20 + (j // 2) * (cell + pad)
            d.text((x0 + 4, y + 10), "".join(novel[j]), fill=(235, 235, 235), font=font)
            for r_i, sp in enumerate(tiles):
                for q in range(args.n):
                    im = Image.fromarray(sp[j * args.n + q], "RGBA").resize((cell, cell), Image.NEAREST)
                    sheet.paste(im, (x0 + 84 + r_i * block + q * (cell + pad), y), im)
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.sheet)
        print(f"sheet: {args.sheet}")


if __name__ == "__main__":
    main()
