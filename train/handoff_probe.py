"""两个网络在一次采样里接力或平均，看能不能一张图同时拿到 Jang 的"认名字"和 Mo 的"稳"。

方案：mo / jang 单独画；mo>jang@p 表示前 p 比例的步数（噪声大、定轮廓的阶段）用 Mo，其余用 Jang；jang>mo@p 反过来；avg 是每步两个预测取平均。
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\data\StarItem")
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))
from colors import NAMES, pixel_colors, with_color  # noqa: E402
from diffusion import T, cosine_alphas_cumprod  # noqa: E402
from eval_bind import embed  # noqa: E402
from generate import load_concepts  # noqa: E402
from model import UNet  # noqa: E402
from pilot import Clip  # noqa: E402
from train import TextEncoder, TokenMap, autocast, to_rgba_uint8  # noqa: E402

dev = torch.device("cuda")
ac = cosine_alphas_cumprod().to(dev)
text = TextEncoder("intfloat/multilingual-e5-small", 32, dev)
nets = {}
for key, run in (("mo", "sic16"), ("jang", "sic17")):
    folder = ROOT / "runs" / run
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(folder / "ema_150000.pt", map_location=dev, weights_only=True))
    nets[key] = (model, TokenMap.load(folder / "tokens.json") if cfg["model"].get("num_tokens") else None)
PLANS = ["mo", "jang", "mo>jang@0.3", "mo>jang@0.5", "jang>mo@0.3", "jang>mo@0.5", "avg"]


@torch.no_grad()
def sample(plan, prompts, n, source, seed=0, steps=30, guidance=5.5):
    low = [p.lower() for p in prompts]
    ctx, mask = text.encode(low)
    nc, nm = text.encode([""])
    tm = nets["jang"][1]
    tok, nt = tm.encode(text.tok, low, 32, dev), tm.encode(text.tok, [""], 32, dev)
    ts = torch.linspace(T - 1, 0, steps, device=dev).round().long()
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, len(prompts), 64):
        c, m, tk = (x[i:i + 64].repeat_interleave(n, 0) for x in (ctx, mask, tok))
        b = c.shape[0]
        x = torch.randn(b, 4, 16, 16, device=dev, generator=g)
        c2, m2 = torch.cat([c, nc.expand(b, -1, -1)]), torch.cat([m, nm.expand(b, -1)])
        tk2 = torch.cat([tk, nt.expand(b, -1)])
        sidx = torch.zeros(2 * b, device=dev, dtype=torch.long)
        src = torch.full((2 * b,), float(source), device=dev)

        def predict(key, t):
            model = nets[key][0]
            tt = t.expand(2 * b)
            with autocast(dev):
                v = model(torch.cat([x, x]), tt, c2, m2, sidx, src, tk2) if key == "jang" else model(torch.cat([x, x]), tt, c2, m2, sidx, src)
            vc, vu = v.float().chunk(2)
            return vu + guidance * (vc - vu)

        for k, t in enumerate(ts):
            if plan in nets:
                v = predict(plan, t)
            elif plan == "avg":
                v = 0.5 * (predict("mo", t) + predict("jang", t))
            else:
                order, p = plan.split("@")
                first, second = order.split(">")
                v = predict(first if k < round(float(p) * steps) else second, t)
            a, s = ac[t].sqrt(), (1 - ac[t]).sqrt()
            x0 = (a * x - s * v).clamp(-1, 1)
            if k == steps - 1:
                x = x0
                break
            eps = (x - a * x0) / s
            ap = ac[ts[k + 1]]
            x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
        out += list(to_rgba_uint8(x))
    return out


concepts = load_concepts()
zh_of = {c["en"]: c["zh"] for c in concepts}
rows = []
for ds in ("synth2", "many", "manual"):
    for line in (ROOT / "dataset" / ds / "metadata.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["name_en"] in zh_of:
            rows.append(r["name_en"])
names = sorted(set(rows))
index = {n: i for i, n in enumerate(names)}
feats = torch.from_numpy(np.load(ROOT / "runs" / "_bind_cache.npz")["feats"]).to(dev)
owner = torch.tensor([index[n] for n in rows], device=dev)
sums = torch.zeros(len(names), feats.shape[1], device=dev).index_add_(0, owner, feats)
centers = sums / sums.norm(dim=-1, keepdim=True)
clip = Clip(dev, ["x"])
rng = random.Random(2)
N = 4
plain = rng.sample(names, 500)
colors = ["red", "blue", "green", "yellow", "purple", "white", "black", "orange"]
colored = []
for n in rng.sample(names, 250):
    c = rng.choice(colors)
    zh = with_color(zh_of[n], c, True)
    if zh:
        colored.append((n, c, f"{zh}, {zh_of[n]}, {n}"))
wands = ["闪电法杖", "法杖", "翡翠魔杖", "末影魔杖", "闪电魔杖", "彩虹法杖", "猫头鹰法杖", "龙骨法杖", "雷电法杖", "星星魔杖"]
combos = ["翡翠镐", "彩虹靴子", "骨头锄", "月亮护符", "铜钥匙", "龙骨斧", "火焰剑", "钻石头盔"]


def ranks(sprites, own):
    sims = embed(clip, sprites) @ centers.T
    mine = sims[torch.arange(len(sims)), own]
    return ((sims > mine[:, None]).sum(1) + 1).cpu().numpy()


print(f"{'plan':13} | exact names: top1, within 10 | color+name: item ok, color ok, both | staffs over-filled")
tiles = {}
for plan in PLANS:
    sp = sample(plan, [f"{zh_of[n]}, {n}" for n in plain], N, 1.0)
    r = ranks(sp, torch.tensor([index[n] for n in plain], device=dev).repeat_interleave(N))
    sp2 = sample(plan, [p for _, _, p in colored], N, 1.0)
    r2 = ranks(sp2, torch.tensor([index[n] for n, _, _ in colored], device=dev).repeat_interleave(N))
    share = []
    for s, w in zip(sp2, np.repeat([NAMES.index(c) for _, c, _ in colored], N)):
        op = s[..., 3] > 0
        px = pixel_colors(s[op][:, :3]) if op.sum() >= 12 else np.array([], dtype=int)
        keep = px != NAMES.index("black") if NAMES[w] != "black" else np.ones(len(px), bool)
        share.append(float((px[keep] == w).mean()) if keep.sum() else 0.0)
    share = np.array(share)
    sp3 = sample(plan, wands, 32, 0.0, seed=11)
    fill = np.array([(s[..., 3] > 0).mean() for s in sp3])
    tiles[plan] = (sp3, sample(plan, combos, 8, 0.0, seed=5))
    print(f"{plan:13} | {np.mean(r <= 1):5.0%} {np.mean(r <= 10):5.0%}          | {np.mean(r2 <= 10):5.0%} {np.mean(share >= 0.3):5.0%} "
          f"{np.mean((r2 <= 10) & (share >= 0.3)):5.0%}        | {np.mean(fill > 0.4):5.0%}", flush=True)

cell, pad = 30, 2
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
show = ["mo", "jang", "mo>jang@0.3", "jang>mo@0.3", "avg"]
labels = wands[:6] + combos
block = len(show) * (cell + pad) + 6
sheet = Image.new("RGB", (90 + 16 * (cell + pad), len(labels) * block), (58, 58, 68))
d = ImageDraw.Draw(sheet)
for i, name in enumerate(labels):
    for k, plan in enumerate(show):
        sp3, sp4 = tiles[plan]
        imgs = sp3[i * 32:i * 32 + 16] if i < 6 else (sp4[(i - 6) * 8:(i - 6) * 8 + 8])
        y = i * block + k * (cell + pad)
        d.text((2, y + 8), f"{name[:4]} {plan[:9]}" if k == 0 else f"     {plan[:11]}", fill=(235, 235, 235), font=font)
        for q, im in enumerate(imgs):
            tile = Image.fromarray(im, "RGBA").resize((cell, cell), Image.NEAREST)
            sheet.paste(tile, (90 + q * (cell + pad), y), tile)
sheet.save(sys.argv[1])
