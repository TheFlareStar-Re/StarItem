"""试验：给小模型一张参考图（缩成 16x16 当底稿），加噪到中途再去噪，看能不能既保住参考图的样子又画成干净的贴图。"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "skill" / "staritem-sprites" / "scripts"))
import generate as G  # noqa: E402

OUT = ROOT / "export"
model = G.Model(ROOT / "export" / "SIC-1.0")
STEPS, GUIDANCE = 30, 5.5


def shrink(ref):
    """参考图直接缩到 16x16：物体裁出来占 15 格，按面积平均。这是最朴素的缩法，结果会发糊。"""
    a = np.asarray(ref.convert("RGBA")).astype(np.float32)
    ys, xs = np.where(a[..., 3] > 127)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 4), dtype=np.float32)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    pre = canvas.copy()
    pre[..., :3] *= canvas[..., 3:] / 255.0
    small = np.asarray(Image.fromarray(pre.astype(np.uint8), "RGBA").resize((15, 15), Image.BOX)).astype(np.float32)
    out = np.zeros((16, 16, 4), dtype=np.uint8)
    alpha = small[..., 3:] / 255.0
    rgb = np.where(alpha > 0, small[..., :3] / np.maximum(alpha, 1e-6), 0)
    opaque = alpha[..., 0] >= 0.5
    out[:15, :15, :3] = np.where(opaque[..., None], np.clip(rgb, 0, 255), 0)
    out[:15, :15, 3] = np.where(opaque, 255, 0)
    return out


def redraw(init, prompt, strength, count, seed, style=None):
    """从参考底稿出发的 DDIM：strength 越大改动越多，1.0 等于完全不看底稿。"""
    text = prompt.strip().lower()
    expanded, known = G.expand(text, model.aliases)
    if style is None:
        style = known if known is not None else 0.0
    ctx, mask = model.encode(expanded)
    null_ctx, null_mask = model.encode("")
    x0 = init.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    x0[:3, init[..., 3] == 0] = -1.0
    total = model.manifest["diffusion"]["timesteps"]
    ts = np.round(np.linspace(total - 1, 0, STEPS)).astype(np.int64)
    first = min(STEPS - 1, int(round((1 - strength) * STEPS)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((count, 4, 16, 16)).astype(np.float32)
    a0 = model.alphas[ts[first]]
    x = (np.sqrt(a0) * x0[None] + np.sqrt(1 - a0) * noise).astype(np.float32) if strength < 1 else noise
    feed = {"ctx": np.concatenate([np.repeat(ctx, count, 0), np.repeat(null_ctx, count, 0)]),
            "mask": np.concatenate([np.repeat(mask, count, 0), np.repeat(null_mask, count, 0)]),
            "size_idx": np.zeros(2 * count, dtype=np.int64), "src_mix": np.full(2 * count, style, dtype=np.float32)}
    for i in range(first, STEPS):
        t = ts[i]
        v = model.unet.run(None, {"x": np.concatenate([x, x]), "t": np.full(2 * count, t, dtype=np.int64), **feed})[0]
        v = v[count:] + GUIDANCE * (v[:count] - v[count:])
        a, s = np.sqrt(model.alphas[t]), np.sqrt(1 - model.alphas[t])
        pred = np.clip(a * x - s * v, -1, 1)
        if i == STEPS - 1:
            x = pred.astype(np.float32)
            break
        eps = (x - a * pred) / s
        nxt = model.alphas[ts[i + 1]]
        x = (np.sqrt(nxt) * pred + np.sqrt(1 - nxt) * eps).astype(np.float32)
    rgba = np.clip((x.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).round().astype(np.uint8)
    opaque = rgba[..., 3] >= 128
    rgba[..., 3] = np.where(opaque, 255, 0)
    rgba[~opaque] = 0
    return list(rgba)


records = [json.loads(line) for line in (ROOT / "dataset" / "synth_raw" / "records.jsonl").open(encoding="utf-8")]
names = ["power bank", "air fryer", "fire extinguisher", "hair dryer", "roller skate", "watering can", "spellbook", "acoustic guitar"]
strengths = [0.35, 0.5, 0.65, 1.0]
cell, pad = 96, 6
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
sheet = Image.new("RGB", (150 + (2 + len(strengths) * 2) * (cell + pad) + 40, len(names) * (cell + pad) + 26), (58, 58, 68))
d = ImageDraw.Draw(sheet)
heads = ["参考图", "直接缩小"] + [f"强度 {s}" if s < 1 else "不看参考图" for s in strengths]
xs = [150, 150 + cell + pad] + [150 + 2 * (cell + pad) + 12 + i * (2 * (cell + pad) + 6) for i in range(len(strengths))]
for x, h in zip(xs, heads):
    d.text((x, 4), h, fill=(255, 220, 120), font=font)
for row, name in enumerate(names):
    r = next(r for r in records if r["en"] == name and r["file"])
    ref = Image.open(ROOT / "dataset" / "synth_raw" / "cut" / r["file"]).convert("RGBA")
    init = shrink(ref)
    y = 26 + row * (cell + pad)
    d.text((4, y + 30), r["zh"], fill=(235, 235, 235), font=font)
    d.text((4, y + 50), name, fill=(170, 170, 170), font=font)
    im = ref.resize((cell, cell), Image.NEAREST)
    sheet.paste(im, (xs[0], y), im)
    im = Image.fromarray(init, "RGBA").resize((cell, cell), Image.NEAREST)
    sheet.paste(im, (xs[1], y), im)
    for si, s in enumerate(strengths):
        for k, sp in enumerate(redraw(init, r["zh"], s, 2, 40 + row)):
            im = Image.fromarray(sp, "RGBA").resize((cell, cell), Image.NEAREST)
            sheet.paste(im, (xs[2 + si] + k * (cell + pad), y), im)
    print(name, flush=True)
sheet.save(OUT / "ref_probe.png")
print(OUT / "ref_probe.png", sheet.size)
