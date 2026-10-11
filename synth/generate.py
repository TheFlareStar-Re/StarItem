"""用 SDXL + 像素画 LoRA 给物品清单里的每个物品生成贴图，缩成 16x16。可中断，重跑会接着做。

Reads synth/concepts/*.tsv (en<TAB>zh<TAB>aliases), writes dataset/synth_raw/:
  cut/<id>_<k>.png     the generated art with background removed (96x96 at the default resolution)
  sprite/<id>_<k>.png  the 16x16 sprite
  records.jsonl        one line per generated image

Usage:
  .venv/Scripts/python.exe -X utf8 synth/generate.py --hours 2.5
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import load_pipeline  # noqa: E402
from sprites import remove_background, to_blocks, to_sprite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "synth_raw"
# 试跑里选定的设置：单个物体、白底；guidance 1.0 不做 CFG，速度快一倍，画面差别不大
PROMPT = "pixel, a single {thing}, one object only, centered, white background"


def load_concepts():
    concepts, seen_en, seen_zh = [], set(), set()
    for path in sorted((ROOT / "synth" / "concepts").glob("*.tsv")):
        category = path.stem.split("_", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0].strip() or not parts[1].strip():
                continue
            en, zh = parts[0].strip().lower(), parts[1].strip()
            if en in seen_en or zh in seen_zh:
                continue  # 不同门类之间重复的只留第一个
            seen_en.add(en)
            seen_zh.add(zh)
            aliases = [a.strip() for a in parts[2].split("|") if a.strip()]
            concepts.append({"id": len(concepts), "en": en, "zh": zh, "aliases": aliases, "category": category})
    return concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=2, help="images per concept")
    ap.add_argument("--batch", type=int, default=2, help="concepts per pipeline call")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = run to the end)")
    ap.add_argument("--limit", type=int, default=0, help="only this many concepts (for testing)")
    args = ap.parse_args()

    concepts = load_concepts()
    # 打乱顺序：中途停下时各门类的覆盖是均匀的
    random.Random(0).shuffle(concepts)
    if args.limit:
        concepts = concepts[:args.limit]
    (OUT / "cut").mkdir(parents=True, exist_ok=True)
    (OUT / "sprite").mkdir(parents=True, exist_ok=True)
    records_path = OUT / "records.jsonl"
    done = set()
    if records_path.exists():
        done = {json.loads(line)["id"] for line in records_path.open(encoding="utf-8")}
    todo = [c for c in concepts if c["id"] not in done]
    print(f"concepts: {len(concepts)}, already done: {len(concepts) - len(todo)}, to do: {len(todo)}", flush=True)
    if not todo:
        return

    device = torch.device("cuda")
    pipe = load_pipeline(device)
    start = time.time()
    made = usable = 0
    with records_path.open("a", encoding="utf-8", newline="\n") as log:
        for i in range(0, len(todo), args.batch):
            if args.hours and time.time() - start > args.hours * 3600:
                print("time limit reached", flush=True)
                break
            group = todo[i:i + args.batch]
            prompts = [PROMPT.format(thing=c["en"]) for c in group for _ in range(args.per)]
            generator = torch.Generator(device=device).manual_seed(1000 + group[0]["id"])
            images = pipe(prompt=prompts, num_inference_steps=args.steps, guidance_scale=1.0, height=args.res,
                          width=args.res, generator=generator).images
            for n, image in enumerate(images):
                concept, k = group[n // args.per], n % args.per
                blocks = to_blocks(image)
                mask = remove_background(blocks)
                sprite, problem = to_sprite(blocks, mask)
                name = f"{concept['id']:05d}_{k}.png"
                if sprite is not None:
                    cut = np.dstack([blocks, np.where(mask, 255, 0).astype(np.uint8)])
                    Image.fromarray(cut, "RGBA").save(OUT / "cut" / name)
                    Image.fromarray(sprite, "RGBA").save(OUT / "sprite" / name)
                    usable += 1
                made += 1
                log.write(json.dumps({**concept, "k": k, "file": name if sprite is not None else None,
                                      "problem": problem}, ensure_ascii=False) + "\n")
            log.flush()
            if (i // args.batch) % 50 == 0:
                elapsed = time.time() - start
                left = (len(todo) - i - len(group)) * args.per * elapsed / max(made, 1)
                print(f"{i + len(group):5}/{len(todo)} concepts, {usable}/{made} usable, "
                      f"{elapsed / max(made, 1):.2f} s/img, about {left / 60:.0f} min left", flush=True)
    print(f"done: {made} images, {usable} usable, {(time.time() - start) / 60:.0f} min")


if __name__ == "__main__":
    main()
