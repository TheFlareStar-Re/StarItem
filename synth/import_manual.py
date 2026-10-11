"""把外部图像模型画的"像素画"大图转成 16x16 贴图。

Reads dataset/manual_raw/<folder>/*.png (folders listed in image25_list*.csv), writes dataset/manual_raw16/:
sprite/<folder>_<n>.png, records.jsonl and a preview sheet.

这类模型画的像素块并不是 16x16：先从图里量出它实际用的网格（通常 30~60 格），按那个网格逐格读成干净的小图，
再把物体裁出来、按"多数颜色"缩到 16x16。

Usage:
  .venv/Scripts/python.exe -X utf8 synth/import_manual.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "manual_raw"
OUT = ROOT / "dataset" / "manual_raw16"


def edge_profile(a, axis):
    """相邻两行（列）之间的颜色跳变总量，像素块的边界处会出现峰值。"""
    d = np.abs(np.diff(a.astype(np.int32), axis=axis)).sum(axis=2)
    return d.sum(axis=1 - axis).astype(np.float64)


def find_grid(a):
    """返回 (格子边长, 起点偏移, 边界能量落在网格线上的比例)。横竖两个方向共用同一个边长。"""
    size = a.shape[0]
    profiles = [edge_profile(a, 0), edge_profile(a, 1)]
    total = sum(p.sum() for p in profiles) + 1e-9
    pos = np.arange(size - 1) + 1.0  # 第 i 个差分对应第 i+1 条像素边界
    best = []
    for period in np.arange(size / 72, size / 12, 0.05):
        captured, offsets = 0.0, []
        for p in profiles:
            # 每条边界到最近网格线的距离不超过 1.5 像素就算落在线上；偏移取能量最大的那个
            phase = pos % period
            hist = np.bincount(np.minimum((phase * 4).astype(int), int(period * 4) - 1), weights=p, minlength=int(period * 4))
            win = np.convolve(np.concatenate([hist, hist[:12]]), np.ones(12), mode="valid")[:len(hist)]  # 3 像素宽的窗口
            k = int(win.argmax())
            captured += win[k]
            offsets.append((k / 4 + 1.5) % period)
        best.append((captured / total, period, offsets))
    top = max(b[0] for b in best)
    # 周期减半时网格线只多不少，得分一样高；取得分接近最高的里面周期最大的，才是真正的格子边长
    score, period, offsets = max((b for b in best if b[0] >= 0.92 * top), key=lambda b: b[1])
    return period, offsets, score


def read_native(a, period, offsets):
    """按量出来的网格逐格取中位色，得到模型实际画的那张小图。"""
    size = a.shape[0]
    ys = np.arange(offsets[0] % period - period, size, period)
    xs = np.arange(offsets[1] % period - period, size, period)
    rows = [(max(0, round(y + period * 0.25)), min(size, round(y + period * 0.75))) for y in ys]
    cols = [(max(0, round(x + period * 0.25)), min(size, round(x + period * 0.75))) for x in xs]
    rows = [r for r in rows if r[1] - r[0] >= 2]
    cols = [c for c in cols if c[1] - c[0] >= 2]
    out = np.zeros((len(rows), len(cols), 3), dtype=np.uint8)
    for i, (ya, yb) in enumerate(rows):
        for j, (xa, xb) in enumerate(cols):
            out[i, j] = np.median(a[ya:yb, xa:xb].reshape(-1, 3), axis=0)
    return out


def cut_background(native, tolerance=60):
    """背景色取四周一圈里最常见的颜色，从边缘向内连通地去掉。返回前景 mask。"""
    h, w = native.shape[:2]
    c = native.astype(np.int32)
    border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    keys = (border[:, 0] >> 4 << 8) | (border[:, 1] >> 4 << 4) | (border[:, 2] >> 4)
    values, counts = np.unique(keys, return_counts=True)
    background = border[keys == values[counts.argmax()]].mean(axis=0)
    close = np.abs(c - background).max(axis=-1) < tolerance
    seen = np.zeros((h, w), dtype=bool)
    stack = [(y, x) for y in range(h) for x in (0, w - 1) if close[y, x]] + [(y, x) for x in range(w) for y in (0, h - 1) if close[y, x]]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    # 物体内部被围住、颜色和背景几乎一样的格子（环、把手中间）也算背景
    seen |= np.abs(c - background).max(axis=-1) < tolerance / 3
    return ~seen


def shrink(native, mask, size=16, fit=15, rim=0.4):
    """把原生网格的小图缩到 size x size。

    直接按多数颜色缩的话，1 像素宽的深色描边会在每个 2x2 块里占一半，细的物体整个变成黑的。
    这里把描边（贴着透明、且明显偏暗的像素）和内部颜色分开：每格的颜色只在内部像素里投票，
    缩完以后再把贴着透明的那一圈按 rim 的比例压暗，保留"有描边"的观感。
    """
    h, w = mask.shape
    coverage = mask.mean()
    if coverage < 0.03:
        return None, "empty"
    if coverage > 0.85:
        return None, "no_background"
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y0 == 0) + (x0 == 0) + (y1 == h) + (x1 == w) >= 2:
        return None, "cut_off"
    rgb, alpha = native[y0:y1, x0:x1].astype(np.int32), mask[y0:y1, x0:x1]
    ch, cw = alpha.shape
    padded = np.pad(alpha, 1)
    near_edge = ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    inner = alpha & ~near_edge
    dark = luma < (np.median(luma[inner]) * 0.6 if inner.any() else 60)
    outline = alpha & near_edge & dark
    scale = fit / max(ch, cw)
    th, tw = max(1, round(ch * scale)), max(1, round(cw * scale))
    out = np.zeros((size, size, 4), dtype=np.uint8)
    oy, ox = (size - th) // 2, (size - tw) // 2
    for ty in range(th):
        ya, yb = ty * ch // th, max((ty + 1) * ch // th, ty * ch // th + 1)
        for tx in range(tw):
            xa, xb = tx * cw // tw, max((tx + 1) * cw // tw, tx * cw // tw + 1)
            cell_a = alpha[ya:yb, xa:xb]
            if cell_a.mean() < 0.5:
                continue
            fill = cell_a & ~outline[ya:yb, xa:xb]
            pixels = rgb[ya:yb, xa:xb][fill if fill.any() else cell_a]
            keys = ((pixels[:, 0] >> 4) << 8) | ((pixels[:, 1] >> 4) << 4) | (pixels[:, 2] >> 4)
            values, counts = np.unique(keys, return_counts=True)
            out[oy + ty, ox + tx, :3] = pixels[keys == values[counts.argmax()]].mean(axis=0).round()
            out[oy + ty, ox + tx, 3] = 255
    opaque = out[..., 3] > 0
    if opaque.sum() < 20:
        return None, "too_small"
    if outline.any() and rim > 0:
        line = rgb[outline].mean(axis=0)
        padded = np.pad(opaque, 1)
        edge = opaque & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
        out[edge, :3] = ((1 - rim) * out[edge, :3] + rim * line).round().astype(np.uint8)
    return out, None


def convert(path):
    """返回 (16x16 RGBA 或 None, 原生网格小图 RGBA, 说明信息)。"""
    a = np.asarray(Image.open(path).convert("RGB"))
    side = min(a.shape[:2])
    a = a[:side, :side]
    period, offsets, score = find_grid(a)
    native = read_native(a, period, offsets)
    mask = cut_background(native)
    rgba = np.dstack([native, np.where(mask, 255, 0).astype(np.uint8)])
    info = {"grid": native.shape[0], "period": round(float(period), 2), "grid_score": round(float(score), 3)}
    # 网格量得不准也没关系：按更细的网格读出来仍是同一张图，后面反正要缩到 16x16
    sprite, problem = shrink(native, mask)
    return sprite, rgba, {**info, "problem": problem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--preview", type=int, default=240, help="at most this many images on the preview sheet, evenly sampled")
    args = ap.parse_args()

    # 各批清单都读进来，后面的批次可以只写自己的 csv
    names = {r["folder"]: r for path in sorted(args.raw.glob("image25_list*.csv"))
             for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    (args.out / "sprite").mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.raw.iterdir() if p.is_dir() and p.name in names)
    records, tiles = [], []
    for folder in folders:
        for n, path in enumerate(sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))):
            sprite, native, info = convert(path)
            name = f"{folder.name}_{n}.png"
            if sprite is not None:
                Image.fromarray(sprite, "RGBA").save(args.out / "sprite" / name)
            row = names[folder.name]
            records.append({"id": int(folder.name[:5]), "en": row["name_en"], "zh": row["name_zh"], "category": row["category"],
                            "k": n, "file": name if sprite is not None else None, "source": str(path.relative_to(args.raw)), **info})
            # 预览只留缩略图，几千张原图全放内存里放不下
            tiles.append((Image.open(path).convert("RGB").resize((128, 128), Image.LANCZOS), native, sprite))
            print(f"{folder.name} {path.name}: grid {info['grid']} (block {info['period']} px, score {info['grid_score']}) "
                  f"-> {info['problem'] or 'ok'}", flush=True)
    with (args.out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 预览：每张图三格，依次是原图、按原生网格读出的小图、16x16 结果
    cell, per_row = 128, 3
    if len(tiles) > args.preview:
        step = len(tiles) / args.preview
        tiles = [tiles[int(i * step)] for i in range(args.preview)]
    sheet = Image.new("RGB", (per_row * (3 * cell + 12), -(-len(tiles) // per_row) * (cell + 6)), (58, 58, 68))
    for i, (orig, native, sprite) in enumerate(tiles):
        x, y = (i % per_row) * (3 * cell + 12), (i // per_row) * (cell + 6)
        sheet.paste(orig, (x, y))
        im = Image.fromarray(native, "RGBA").resize((cell, cell), Image.NEAREST)
        sheet.paste(im, (x + cell, y), im)
        if sprite is not None:
            im = Image.fromarray(sprite, "RGBA").resize((cell, cell), Image.NEAREST)
            sheet.paste(im, (x + 2 * cell, y), im)
    sheet.save(args.out / "preview.png")
    usable = sum(r["file"] is not None for r in records)
    grids = [r["grid"] for r in records]
    print(f"\n{len(records)} images, {usable} converted; native grid: min {min(grids)}, median {int(np.median(grids))}, max {max(grids)}")
    print(f"preview: {args.out / 'preview.png'}")


if __name__ == "__main__":
    main()
