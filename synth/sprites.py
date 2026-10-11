"""把大模型生成的像素画变成 16x16 的物品贴图：缩小、抠背景、裁切、再缩小。"""
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

BLOCK = 8  # pixel-art-xl 在 1024 分辨率下每个色块是 8x8


def to_blocks(image):
    """1024x1024 的 PIL 图 -> 128x128 的 uint8 RGB，每个色块取中心像素。"""
    a = np.asarray(image.convert("RGB"))
    return a[BLOCK // 2::BLOCK, BLOCK // 2::BLOCK].copy()


def components(mask):
    """4 连通分量。返回 (标号图, 各分量的像素数)，标号从 1 开始，0 表示不在 mask 里。"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = [0]
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        label = len(sizes)
        labels[sy, sx] = label
        queue = deque([(sy, sx)])
        count = 0
        while queue:
            y, x = queue.popleft()
            count += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        sizes.append(count)
    return labels, np.array(sizes)


def remove_background(rgb, tolerance=30):
    """找出图里的物体，返回 bool 的前景 mask。

    背景取全图最常见的颜色、最大的那一片；不碰到图像边缘的非背景连通块才算物体，
    这样生成图里自带的边框、卡片底、边框外的区域都会被丢掉。
    """
    h, w, _ = rgb.shape
    # 颜色量化到 5 bit 后找出现最多的一种，当作背景色
    q = (rgb >> 3).astype(np.int32)
    keys = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    mode = np.bincount(keys.ravel()).argmax()
    bg = rgb[keys == mode].astype(np.int16).mean(axis=0)
    dist = np.abs(rgb.astype(np.int16) - bg).max(axis=-1)
    close = dist < tolerance
    if bg.min() > 200:
        # 白底上的浅灰投影也当背景；物体有深色描边时不会被连带吃掉
        value = rgb.max(axis=-1).astype(np.int16)
        close |= (value - rgb.min(axis=-1) < 24) & (value > 175)
    labels, sizes = components(close)
    if len(sizes) < 2 or sizes.max() < 0.15 * h * w:
        return np.ones((h, w), dtype=bool)  # 没有成片的背景
    background = labels == sizes.argmax()

    labels, sizes = components(~background)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    inner = [i for i in range(1, len(sizes)) if i not in border]
    if not inner:
        return np.ones((h, w), dtype=bool)
    largest = max(sizes[i] for i in inner)
    keep = [i for i in inner if sizes[i] >= 0.08 * largest]
    if len(keep) > 3:
        return np.zeros((h, w), dtype=bool)  # 一张图里排了很多个小物件（精灵表），不要
    mask = np.isin(labels, keep)
    # 物体内部和背景颜色完全一样的洞（甜甜圈的孔）保持透明；只是接近背景色的部分（奶泡）保留
    mask &= dist >= 10
    return mask


def remove_shadow(rgb, mask):
    """去掉物体脚下的投影：从物体外沿向内，吃掉相连的中性浅灰像素。有深色描边的物体不受影响。"""
    h, w = mask.shape
    value = rgb.max(axis=-1).astype(np.int16)
    gray = (value - rgb.min(axis=-1) < 14) & (value >= 105) & (value <= 240) & mask
    out = mask.copy()
    queue = deque()
    for y, x in zip(*np.nonzero(gray)):
        # 紧挨着透明区域或图像边缘的灰色像素是起点
        edge = y in (0, h - 1) or x in (0, w - 1) or not (mask[y - 1, x] and mask[y + 1, x] and mask[y, x - 1] and mask[y, x + 1])
        if edge:
            out[y, x] = False
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and out[ny, nx] and gray[ny, nx]:
                out[ny, nx] = False
                queue.append((ny, nx))
    # 灰色的物体整个被吃掉时放弃，保持原样
    return out if out.sum() >= 0.5 * mask.sum() else mask


def to_sprite(rgb, mask, size=16, fit=15):
    """把前景裁出来缩到 size x size（物体最长边占 fit 个像素）。

    每个目标像素取它覆盖范围里出现最多的颜色，而不是平均色，这样缩小后仍是边界分明的色块。
    返回 (uint8 的 RGBA 贴图, 问题说明)；问题说明为 None 表示可用。
    """
    h, w = mask.shape
    coverage = mask.mean()
    if coverage < 0.04:
        return None, "empty"
    if coverage > 0.85:
        return None, "no_background"
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    touching = (y0 == 0) + (x0 == 0) + (y1 == h) + (x1 == w)
    if touching >= 2:
        return None, "cut_off"
    crop_rgb, crop_a = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ch, cw = crop_a.shape
    scale = fit / max(ch, cw)
    th, tw = max(1, round(ch * scale)), max(1, round(cw * scale))
    out = np.zeros((size, size, 4), dtype=np.uint8)
    oy, ox = (size - th) // 2, (size - tw) // 2
    for ty in range(th):
        ya, yb = ty * ch // th, max((ty + 1) * ch // th, ty * ch // th + 1)
        for tx in range(tw):
            xa, xb = tx * cw // tw, max((tx + 1) * cw // tw, tx * cw // tw + 1)
            cell_a = crop_a[ya:yb, xa:xb]
            if cell_a.mean() < 0.5:
                continue
            pixels = crop_rgb[ya:yb, xa:xb][cell_a].astype(np.int32)
            # 量化到每通道 4 bit 后投票，取得票最多的那一组的平均色
            keys = ((pixels[:, 0] >> 4) << 8) | ((pixels[:, 1] >> 4) << 4) | (pixels[:, 2] >> 4)
            values, counts = np.unique(keys, return_counts=True)
            winner = pixels[keys == values[counts.argmax()]].mean(axis=0)
            out[oy + ty, ox + tx] = (*winner.round().astype(np.uint8), 255)
    if (out[..., 3] > 0).sum() < 20:
        return None, "too_small"
    return out, None
