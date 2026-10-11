"""写第三批 image2.5 素材清单：训练图不足 10 张的物品，按新模型画得好不好分两档。

Reads the per-concept recall CSV written by train/eval_recall.py --per-concept and the training datasets,
writes dataset/manual_raw/image25_list_batch3.csv in the same format as the earlier lists.
Tier 3A: fewer than 10 training sprites and the model's median rank is above 300. Tier 3B: the rest under 10.
Tier 3C: items whose name already appears on 5 or more hand-drawn mod sprites, so drafts are probably not needed.

Usage:
  python -X utf8 tools/write_image25_batch3.py --ranks export/dev/sic15_per_concept.csv
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from generate import load_concepts  # noqa: E402

PROMPT = ("A 16x16 pixel art game item icon of {name}, in the style of a Minecraft item texture. One single object, centered, "
          "filling most of the canvas, with a dark 1-pixel outline and 3 to 5 flat shades per color. Exactly 16 by 16 large square "
          "pixels; every pixel is a uniform solid block with hard edges, perfectly aligned to the grid. Solid flat {bg} background "
          "and nothing else in the image: no shadow, no text, no border, no grid lines, no anti-aliasing, no gradients, "
          "no {avoid} on the object itself.")
# 物品本身偏粉/紫时改用绿色背景，免得抠背景时把物品一起抠掉
PINKISH = re.compile(r"\b(pink|purple|magenta|violet|lavender|lilac|fuchsia|orchid|amethyst|rose|peony|flamingo|"
                     r"cherry blossom|sakura|plum|grape|grapes|eggplant|beet|beetroot|dragon fruit|taro|ube|lotus flower)\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=["synth2", "many", "manual"])
    ap.add_argument("--min-sprites", type=int, default=10)
    ap.add_argument("--bad-rank", type=int, default=300)
    ap.add_argument("--hand-drawn", type=int, default=5, help="hand-drawn sprites with the same name that make drafts unnecessary")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "manual_raw" / "image25_list_batch3.csv")
    args = ap.parse_args()

    count = collections.Counter()
    for d in args.datasets:
        count.update(json.loads(line)["name_en"] for line in (ROOT / "dataset" / d / "metadata.jsonl").open(encoding="utf-8"))
    ranks = {r["name_en"]: r for r in csv.DictReader(args.ranks.open(encoding="utf-8-sig"))}
    clean = [json.loads(line) for line in (ROOT / "dataset" / "clean" / "metadata.jsonl").open(encoding="utf-8")]
    clean_en = "\n".join((c.get("name_en") or "").lower() for c in clean)
    clean_zh = [c.get("name_zh") or "" for c in clean]
    rows = []
    for c in load_concepts():
        n = count[c["en"]]
        if n >= args.min_sprites:
            continue
        median, best = int(ranks[c["en"]]["median_rank"]), int(ranks[c["en"]]["best_rank"])
        pink = bool(PINKISH.search(c["en"].lower()))
        # 模组手绘贴图里名字带这个词的数量：英文按整词、中文按包含
        word = re.compile(r"(?<![a-z])" + re.escape(c["en"].lower()) + r"s?(?![a-z])")
        drawn = max(len(word.findall(clean_en)), sum(1 for z in clean_zh if c["zh"] and c["zh"] in z))
        tier = "3C" if drawn >= args.hand_drawn else "3A" if median > args.bad_rank else "3B"
        slug = re.sub(r"[^a-z0-9]+", "_", c["en"].lower()).strip("_")
        rows.append({
            "tier": tier, "folder": f"{int(c['id']):05d}_{slug}", "name_en": c["en"],
            "name_zh": c["zh"], "category": c["category"],
            "evidence": f"训练图 {n} 张；新模型 4 张的排名中位数 {median}、最好 {best}"
                        + (f"；模组手绘贴图里有 {drawn} 张同名" if drawn else ""), "images": 3,
            "prompt": PROMPT.format(name=c["en"], bg="bright green (#00FF00)" if pink else "magenta (#FF00FF)",
                                    avoid="green" if pink else "magenta or pink"),
            "_n": n, "_median": median,
        })
    rows.sort(key=lambda r: (r["tier"], r["_n"], -r["_median"]))
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "folder", "name_en", "name_zh", "category", "evidence", "images", "prompt"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tiers = collections.Counter(r["tier"] for r in rows)
    print(f"{args.out}: {len(rows)} items, {dict(tiers)}, green background for {sum('green' in r['prompt'] for r in rows)}")
    for tier in sorted(tiers):
        cats = collections.Counter(r["category"] for r in rows if r["tier"] == tier)
        print(f"  {tier}: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))


if __name__ == "__main__":
    main()
