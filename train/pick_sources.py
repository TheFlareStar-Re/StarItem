"""给合成数据里的每个物品决定画风：按手绘风格画得像，还是要混入合成图的风格才画得像。

For every concept, sprites are generated with each style mix and scored with CLIP against the concept name.
These concepts only exist in the synthetic data, so the synthetic mix is the default; the hand-drawn style
(mix 0) is chosen only where it scores clearly better (the mod sprites happen to cover the concept).
Writes runs/<run>/concept_sources.json: {"中文名, english name": mix}.

Usage:
  python -X utf8 train/pick_sources.py --run sic11 --step 160000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "synth"))
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from model import UNet  # noqa: E402
from pilot import CLIP_MEAN, CLIP_STD, on_gray  # noqa: E402
from train import TextEncoder, autocast, to_rgba_uint8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sic11")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--mix", type=float, default=0.7, help="the synthetic style mix to compare against 0")
    ap.add_argument("--margin", type=float, default=0.01, help="how much better mix 0 must score to be chosen")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=5.5)
    args = ap.parse_args()

    from transformers import AutoTokenizer, CLIPModel

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda")
    concepts = {}
    for folder in cfg["args"]["extra"]:
        for line in (Path(folder) / "metadata.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            concepts.setdefault(f"{r['name_zh']}, {r['name_en']}".lower(), r["name_en"])
    pairs = list(concepts)
    print(f"{len(pairs)} concepts", flush=True)

    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    ctx_all, mask_all = text.encode(pairs)
    null_ctx, null_mask = text.encode([""])
    del text
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(run / f"ema_{args.step:06d}.pt", map_location=dev, weights_only=True))
    ac = cosine_alphas_cumprod().to(dev)

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    norm = lambda o: (lambda f: f / f.norm(dim=-1, keepdim=True))(o if torch.is_tensor(o) else o.pooler_output)  # noqa: E731
    with torch.no_grad():
        feats = []
        names = [f"a pixel art icon of {concepts[p]}" for p in pairs]
        for i in range(0, len(names), 512):
            enc = tok(names[i:i + 512], padding=True, truncation=True, return_tensors="pt").to(dev)
            feats.append(norm(clip.get_text_features(**enc)))
        text_feats = torch.cat(feats)

    scores = {}
    chunk = 256
    for mix in (0.0, args.mix):
        total = torch.zeros(len(pairs), device=dev)
        g = torch.Generator(device=dev).manual_seed(0)
        for i in range(0, len(pairs), chunk):
            ctx = ctx_all[i:i + chunk].repeat_interleave(args.n, 0)
            mask = mask_all[i:i + chunk].repeat_interleave(args.n, 0)
            with autocast(dev):
                x = ddim_sample(model, ctx, mask, null_ctx, null_mask, 16, 0, ac, steps=30, guidance=args.guidance,
                                generator=g, source=mix)
            images = np.stack([on_gray(s, 224) for s in to_rgba_uint8(x)])
            px = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            px = ((px - CLIP_MEAN) / CLIP_STD).to(dev)
            with torch.no_grad():
                img = norm(clip.get_image_features(pixel_values=px))
            own = text_feats[i:i + chunk].repeat_interleave(args.n, 0)
            total[i:i + chunk] = (img * own).sum(-1).view(-1, args.n).mean(1)
        scores[mix] = total.cpu().numpy()
        print(f"mix {mix}: mean CLIP score {scores[mix].mean():.4f}", flush=True)

    hand_drawn = scores[0.0] - scores[args.mix] > args.margin
    # 每个物品记两个数：选定的画风，以及合成风格比手绘风格高出多少分（导出时还会用到）
    diff = scores[args.mix] - scores[0.0]
    result = {p: [0.0 if b else args.mix, round(float(d), 4)] for p, b, d in zip(pairs, hand_drawn, diff)}
    (run / "concept_sources.json").write_text(json.dumps(result, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"hand-drawn style chosen for {int(hand_drawn.sum())} of {len(pairs)} concepts, synthetic mix for the rest")
    for key in ("烤火鸡, roast turkey", "拿铁, cup of latte coffee", "汉堡, hamburger", "寿司, sushi roll"):
        if key in result:
            i = pairs.index(key)
            print(f"  {key}: {scores[0.0][i]:.3f} vs {scores[args.mix][i]:.3f} -> {result[key][0]}")


if __name__ == "__main__":
    main()
