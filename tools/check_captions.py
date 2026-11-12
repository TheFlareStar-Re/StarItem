"""整理云端写回来的贴图描述：核对颜色词，写出 dataset/captions.jsonl。

Incomplete or overlong captions and ones that talk about pixels or the background get "ok": false, so that training
can skip them. Every caption is also checked against the sprite's pixels: a color word naming a color the sprite does
not contain (nor a neighbouring one) sets "color_check": false. That check is only a hint: on a sample of 40 flagged
captions most were fine (a small detail really had that color), so it does not make a caption unusable.

Usage:
  python -X utf8 tools/check_captions.py cloud/pull/captions.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from colors import COLORS_ZH, NAMES, pixel_colors  # noqa: E402

ZH = {v[0]: k for k, v in COLORS_ZH.items()}
ZH.update({"金": "yellow", "银": "gray", "褐": "brown"})
# 小图上相邻的颜色常常分不清，描述说了相邻的颜色不算错
NEAR = {"yellow": ("yellow", "orange", "brown"), "gray": ("gray", "white", "black"), "brown": ("brown", "orange", "red", "yellow"),
        "orange": ("orange", "brown", "red", "yellow"), "pink": ("pink", "red", "purple"), "purple": ("purple", "pink", "blue"),
        "cyan": ("cyan", "blue", "green"), "blue": ("blue", "cyan", "purple"), "green": ("green", "cyan", "yellow"),
        "red": ("red", "pink", "orange", "brown"), "white": ("white", "gray", "yellow"), "black": ("black", "gray", "blue", "brown")}
BANNED = ("像素", "贴图", "背景")


def main():
    names = {}
    for dataset in ("clean", "synth2", "many", "manual"):
        for line in (ROOT / "dataset" / dataset / "metadata.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["size"] == 16:
                names[(dataset, r["file"])] = r.get("name_zh") or ""
    best = {}
    for line in Path(sys.argv[1]).open(encoding="utf-8"):
        r = json.loads(line)
        key = (r["dataset"], r["file"])
        if key not in best or (r.get("caption_zh") and r.get("caption_en")):
            best[key] = r
    why = Counter()
    out = []
    for key, r in best.items():
        zh, en = r.get("caption_zh"), r.get("caption_en")
        reason = None
        if not zh or not en:
            reason = "incomplete"
        elif len(zh) > 40 or len(en.split()) > 30:
            reason = "too long"
        elif any(w in zh for w in BANNED):
            reason = "mentions pixels or background"
        color_ok = True
        if reason is None:
            a = np.asarray(Image.open(ROOT / "dataset" / key[0] / key[1]).convert("RGBA"))
            opaque = a[..., 3] > 0
            share = np.bincount(pixel_colors(a[opaque][:, :3]), minlength=len(NAMES)) / max(opaque.sum(), 1)
            for ch, color in ZH.items():
                # 名字里本来就有的颜色字（"金苹果"）不算描述说的
                if ch in zh and ch not in names.get(key, "") and sum(share[NAMES.index(n)] for n in NEAR[color]) < 0.04:
                    color_ok = False
                    break
        why[reason or ("ok" if color_ok else "ok, color check failed")] += 1
        out.append({"dataset": key[0], "file": key[1], "zh": zh, "en": en, "ok": reason is None, "color_check": color_ok,
                    **({"why": reason} if reason else {})})
    target = ROOT / "dataset" / "captions.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8", newline="\n")
    print(f"{len(out)} captions -> {target}; " + ", ".join(f"{k}: {v} ({v / len(out):.1%})" for k, v in why.most_common()))
    missing = len(names) - len({k for k in best if k in names})
    print(f"training sprites without any caption: {missing}")


if __name__ == "__main__":
    main()
