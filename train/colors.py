"""从贴图像素算出主色，给提示词补上颜色词（"red apple"、"红色的苹果"）。"""
import numpy as np

COLORS_ZH = {"red": "红色", "orange": "橙色", "yellow": "黄色", "green": "绿色", "cyan": "青色", "blue": "蓝色",
             "purple": "紫色", "pink": "粉色", "brown": "棕色", "white": "白色", "gray": "灰色", "black": "黑色"}
NAMES = list(COLORS_ZH)
# 名称里已经带这些字时不再补颜色词
ZH_COLOR_CHARS = "红橙黄绿青蓝紫粉棕褐白灰黑金银"
EN_COLOR_WORDS = set(NAMES) | {"golden", "gold", "silver", "grey", "violet", "crimson", "scarlet", "azure", "lime",
                               "magenta", "teal", "dark", "light"}


def pixel_colors(rgb):
    """(N,3) uint8 -> (N,) 颜色名下标。"""
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(axis=1), c.min(axis=1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    hue = np.zeros_like(mx)
    d = np.maximum(delta, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6.0, hue)
    hue = np.where(mx == g, (b - r) / d + 2.0, hue)
    hue = np.where(mx == b, (r - g) / d + 4.0, hue)
    hue = hue * 60.0

    idx = lambda name: NAMES.index(name)  # noqa: E731
    out = np.full(len(rgb), idx("red"))
    for name, lo, hi in (("orange", 15, 42), ("yellow", 42, 70), ("green", 70, 165), ("cyan", 165, 200),
                         ("blue", 200, 255), ("purple", 255, 295), ("pink", 295, 345)):
        out[(hue >= lo) & (hue < hi)] = idx(name)
    # 暗的橙黄是棕色，暗的粉是紫色
    out[(hue >= 12) & (hue < 50) & ((mx < 0.62) | ((sat < 0.55) & (mx < 0.8)))] = idx("brown")
    out[(hue >= 50) & (hue < 70) & (mx < 0.5)] = idx("brown")
    out[(hue >= 295) & (hue < 345) & (mx < 0.5)] = idx("purple")
    gray = sat < 0.16
    out[gray] = idx("gray")
    out[gray & (mx > 0.82)] = idx("white")
    out[mx < 0.22] = idx("black")
    return out


def dominant_color(rgba, min_share=0.5):
    """(S,S,4) uint8 -> 主色的英文名；没有哪种颜色明显占多数时返回 None。"""
    opaque = rgba[..., 3] > 0
    if opaque.sum() < 12:
        return None
    counts = np.bincount(pixel_colors(rgba[opaque][:, :3]), minlength=len(NAMES)).astype(np.float32)
    black = NAMES.index("black")
    # 描边和阴影多是黑色，除非整张图几乎都是黑的，否则不参与评选
    if counts[black] < 0.7 * counts.sum():
        counts[black] = 0
    if counts.sum() < 12:
        return None
    best = int(counts.argmax())
    return NAMES[best] if counts[best] / counts.sum() >= min_share else None


def with_color(name, color, chinese):
    """给名称加上颜色词；名称里已经有颜色词时返回 None。"""
    if chinese:
        return None if any(ch in name for ch in ZH_COLOR_CHARS) else f"{COLORS_ZH[color]}的{name}"
    return None if EN_COLOR_WORDS & set(name.split()) else f"{color} {name}"
