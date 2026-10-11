"""筛选合成贴图并整理成训练用的数据集 dataset/synth（格式和 dataset/clean 一致）。

Each generated image is checked with CLIP: among all concepts, how high does its own concept rank?
Images that CLIP cannot connect to their concept are dropped.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/build_dataset.py --max-rank 20
  .venv/Scripts/python.exe -X utf8 synth/build_dataset.py --raw dataset/synth2_raw --out dataset/synth2 --direct --max-rank 300
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import CLIP_MEAN, CLIP_STD, Clip, on_gray  # noqa: E402
from sprites import remove_shadow, to_sprite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[一-鿿]")


TEMPLATES = ("a pixel art icon of {}", "{}", "pixel art of {}, video game item sprite")


def direct_ranks(clip, tokenizer, sprites, names, answers):
    """16x16 的贴图给 CLIP 看很吃力，单一视图的排名噪声大。这里把灰底/白底、最近邻/双三次放大
    四种视图和三种提示词写法的相似度加在一起再排名，比单一视图稳定得多。"""
    device = clip.device
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)
    with torch.no_grad():
        text = []
        for template in TEMPLATES:
            texts = [template.format(n) for n in names]
            feats = []
            for i in range(0, len(texts), 512):
                enc = tokenizer(texts[i:i + 512], padding=True, truncation=True, return_tensors="pt").to(device)
                feats.append(clip._norm(clip.model.get_text_features(**enc)))
            text.append(torch.cat(feats))
        text = torch.stack(text).sum(dim=0)
        ranks = []
        for i in range(0, len(sprites), 256):
            sims = 0
            for background in (128.0, 255.0):
                for resample in (Image.NEAREST, Image.BICUBIC):
                    batch = []
                    for rgba in sprites[i:i + 256]:
                        alpha = rgba[..., 3:].astype(np.float32) / 255.0
                        rgb = rgba[..., :3].astype(np.float32) * alpha + background * (1 - alpha)
                        batch.append(np.asarray(Image.fromarray(rgb.round().astype(np.uint8)).resize((224, 224), resample)))
                    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(device) / 255.0
                    sims = sims + clip._norm(clip.model.get_image_features(pixel_values=(x - mean) / std)) @ text.T
            own = sims[torch.arange(len(sims)), torch.tensor(answers[i:i + 256], device=device)]
            ranks += ((sims > own[:, None]).sum(dim=1) + 1).tolist()
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rank", type=int, default=20, help="keep an image if its concept ranks at least this high")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--raw", type=Path, default=ROOT / "dataset" / "synth_raw", help="folder written by generate.py or generate_lora.py")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "synth")
    ap.add_argument("--direct", action="store_true",
                    help="the raw folder already holds final 16x16 sprites (generate_lora.py), nothing to shrink")
    ap.add_argument("--all-concepts", action="store_true", help="rank against every concept in synth/concepts")
    ap.add_argument("--rescue-rank", type=int, default=0,
                    help="for concepts with no sprite kept, keep the single best one if it ranks at least this high")
    ap.add_argument("--max-border", type=int, default=20,
                    help="with --direct: drop sprites with more opaque cells than this on the outer ring (leftover frames)")
    args = ap.parse_args()
    RAW, OUT = args.raw, args.out  # noqa: N806

    records = [json.loads(line) for line in (RAW / "records.jsonl").open(encoding="utf-8")]
    concepts = {r["id"]: r for r in records}
    if args.all_concepts:
        # 只生成了一部分物品时，排名仍然在完整的物品清单里算，筛选标准才和全量生成时一致
        from generate import load_concepts

        concepts = {c["id"]: c for c in load_concepts()}
    ids = sorted(concepts)
    index = {cid: i for i, cid in enumerate(ids)}
    usable = [r for r in records if r["file"]]
    if args.direct:
        # 两台机器各跑过同一个物品时，记录和文件可能对不上，缺文件的跳过
        usable = [r for r in usable if (RAW / "sprite" / r["file"]).exists()]
    print(f"generated {len(records)} images for {len(ids)} concepts, {len(usable)} with a sprite", flush=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    clip = Clip(device, [f"a pixel art icon of {concepts[c]['en']}" for c in ids])
    ranks = []
    if args.direct:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        sprites = [np.asarray(Image.open(RAW / "sprite" / r["file"]).convert("RGBA")) for r in usable]
        ranks = direct_ranks(clip, tokenizer, sprites, [concepts[c]["en"] for c in ids], [index[r["id"]] for r in usable])
    for i in range(0, 0 if args.direct else len(usable), 256):
        chunk = usable[i:i + 256]
        images = []
        for r in chunk:
            cut = np.asarray(Image.open(RAW / "cut" / r["file"]).convert("RGBA"))
            gray = Image.fromarray(on_gray(cut, cut.shape[0]))
            images.append(np.asarray(gray.resize((224, 224), Image.BICUBIC)))
        ranks += clip.rank(images, [index[r["id"]] for r in chunk])
    ranks = np.array(ranks)
    for k in (1, 5, 20, 100, 300):
        print(f"  concept ranked in the top {k:3}: {(ranks <= k).mean():.0%}")

    if OUT.exists():
        shutil.rmtree(OUT)
    kept = Counter()
    dropped = Counter()
    per_concept = Counter()
    rescued = 0
    with_meta = []
    order = np.argsort(ranks, kind="stable")
    # 第二遍只照顾第一遍里一张都没留下的物品：排名放宽，贴边也放宽（键盘这类宽物体本来就贴边），每个物品只留最好的一张
    passes = [(i, False) for i in range(len(usable))] + [(int(i), True) for i in order] * bool(args.rescue_rank)
    for i, rescue in passes:
        r, rank = usable[i], ranks[i]
        if rescue and (per_concept[r["id"]] or rank > args.rescue_rank):
            continue
        if not rescue and rank > args.max_rank:
            continue
        if args.direct:
            sprite, problem = np.asarray(Image.open(RAW / "sprite" / r["file"]).convert("RGBA")), None
            # 手绘贴图很少贴满边缘，贴边格子多的基本是没抠干净的边框或底色
            ring = int((sprite[0, :, 3] > 0).sum() + (sprite[-1, :, 3] > 0).sum()
                       + (sprite[1:-1, 0, 3] > 0).sum() + (sprite[1:-1, -1, 3] > 0).sum())
            if ring > (2 * args.max_border if rescue else args.max_border):
                sprite, problem = None, "touches_border"
        else:
            # 贴图从抠好背景的原图重新缩一遍：去掉投影，按"多数颜色"缩小
            cut = np.asarray(Image.open(RAW / "cut" / r["file"]).convert("RGBA"))
            rgb, mask = cut[..., :3], cut[..., 3] > 0
            sprite, problem = to_sprite(rgb, remove_shadow(rgb, mask))
        if sprite is None:
            dropped[problem] += not rescue
            continue
        rescued += rescue
        rel = f"images/{r['category']}/{r['file']}"
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sprite, "RGBA").save(dest)
        with_meta.append({
            "file": rel, "name_en": r["en"], "name_zh": r["zh"], "name_source": "concept",
            "alt_names_en": [a for a in r["aliases"] if not CJK.search(a)],
            "alt_names_zh": [a for a in r["aliases"] if CJK.search(a)],
            "mod": "synthetic", "slug": "synthetic", "project_id": 0, "namespace": r["category"],
            "item_id": f"{r['id']:05d}_{r['k']}", "kind": "synthetic", "layers": [], "size": 16, "animated": False,
            "license_verdict": "synthetic", "cf_license": "", "clip_rank": int(rank),
            "sha1": hashlib.sha1(sprite.tobytes()).hexdigest(),
        })
        kept[r["category"]] += 1
        per_concept[r["id"]] += 1
    with (OUT / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for m in with_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"kept {len(with_meta)} sprites covering {len(per_concept)} of {len(ids)} concepts; "
          f"rank too low: {int((ranks > args.max_rank).sum())}, unusable after cleanup: {dict(dropped)}, "
          f"rescued as the only sprite of a concept: {rescued}")
    print("per category:", dict(kept.most_common()))
    print(f"dataset: {OUT}")


if __name__ == "__main__":
    main()
