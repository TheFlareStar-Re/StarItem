"""把外部模型的底稿做成训练数据集 dataset/manual（格式和 dataset/synth2 一致）。

For each source image converted by import_manual.py: keep the direct 16x16 conversion, and let our LoRA redraw
it several times (img2img) to get varied sprites in the mod style. Redraws whose silhouette drifted too far
from the source are dropped.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/import_manual.py
  .venv/Scripts/python.exe -X utf8 synth/build_manual.py
  CUDA_VISIBLE_DEVICES=1 python -X utf8 synth/build_manual.py --shard 1/2   # one process per GPU, then:
  python -X utf8 synth/build_manual.py --merge
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import load_concepts  # noqa: E402
from lora_sprites import check, load_pipeline, read_grid  # noqa: E402
from redraw_manual import as_canvas  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[一-鿿]")


def overlap(a, b):
    """两张贴图不透明区域的交并比。"""
    a, b = a[..., 3] > 0, b[..., 3] > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "dataset" / "manual_raw16")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "manual")
    ap.add_argument("--redraws", type=int, default=6, help="redraws per source image")
    ap.add_argument("--strengths", default="0.8,0.88")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--min-overlap", type=float, default=0.4)
    ap.add_argument("--rebuild", action="store_true", help="start over instead of adding only the new source images")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours; run again to continue")
    ap.add_argument("--batch", type=int, default=1, help="source images per pipeline call; raise it on a large GPU")
    ap.add_argument("--shard", default="0/1", help="I/N: do every N-th source image starting at I, writing metadata_shardI.jsonl")
    ap.add_argument("--merge", action="store_true", help="fold the metadata_shard*.jsonl files into metadata.jsonl and stop")
    args = ap.parse_args()
    part, parts = (int(x) for x in args.shard.split("/"))
    shard_files = sorted(args.out.glob("metadata_shard*.jsonl"))

    def write(path, rows, mode="w"):
        with path.open(mode, encoding="utf-8", newline="\n") as f:
            for m in rows:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def summary(rows):
        counts = sorted(Counter(m["name_en"] for m in rows).values())
        return (f"{len(rows)} sprites for {len(counts)} concepts (per concept: min {counts[0]}, "
                f"median {counts[len(counts) // 2]}, max {counts[-1]})")

    if args.merge:
        meta = [json.loads(line) for path in [args.out / "metadata.jsonl", *shard_files] if path.exists()
                for line in path.open(encoding="utf-8")]
        meta = list({m["item_id"]: m for m in meta}.values())
        missing = [m["file"] for m in meta if not (args.out / m["file"]).exists()]
        write(args.out / "metadata.jsonl", meta)
        for path in shard_files:
            path.unlink()
        print(f"merged {len(shard_files)} shard files: {summary(meta)}; files missing on disk: {len(missing)}")
        return

    from diffusers import StableDiffusionXLImg2ImgPipeline

    device = torch.device("cuda")
    pipe, state = load_pipeline(device)
    pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components).to(device)
    pipe.set_progress_bar_config(disable=True)
    strengths = [float(s) for s in args.strengths.split(",")]
    per = -(-args.redraws // len(strengths))

    concepts = {c["en"]: c for c in load_concepts()}
    records = [json.loads(line) for line in (args.src / "records.jsonl").open(encoding="utf-8")]
    records = [r for r in records if r["file"]]
    if args.out.exists() and args.rebuild:
        shutil.rmtree(args.out)
    meta, dropped, per_concept = [], Counter(), Counter()
    # 已经处理过的底稿不重做，只补新的；中途停下后重跑也从这里接上
    if (args.out / "metadata.jsonl").exists():
        meta = [json.loads(line) for line in (args.out / "metadata.jsonl").open(encoding="utf-8")]
        per_concept.update(m["name_en"] for m in meta)
    # 分片跑时，各片先各写各的文件，最后用 --merge 合到一起
    sharded = parts > 1
    shard_path = args.out / f"metadata_shard{part}.jsonl"
    elsewhere = [json.loads(line) for path in shard_files for line in path.open(encoding="utf-8")] if sharded else []
    have = {tuple(m["item_id"].split("_")[:2]) for m in meta + elsewhere}
    records = [r for r in records if (f"{r['id']:05d}", str(r["k"])) not in have]
    if sharded:
        records = records[part::parts]
        args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(have)} source images already done, {len(records)} to do", flush=True)
    started = time.time()
    first_new = written = len(meta)

    def save():
        nonlocal written
        if sharded:
            write(shard_path, meta[written:], "a")
        else:
            write(args.out / "metadata.jsonl", meta)
        written = len(meta)

    def add(r, sprite, tag, kind):
        concept = concepts[r["en"]]
        rel = f"images/{concept['category']}/{r['id']:05d}_{r['k']}_{tag}.png"
        (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sprite, "RGBA").save(args.out / rel)
        meta.append({
            "file": rel, "name_en": r["en"], "name_zh": concept["zh"], "name_source": "concept",
            "alt_names_en": [a for a in concept["aliases"] if not CJK.search(a)],
            "alt_names_zh": [a for a in concept["aliases"] if CJK.search(a)],
            "mod": "synthetic", "slug": "synthetic", "project_id": 0, "namespace": concept["category"],
            "item_id": f"{r['id']:05d}_{r['k']}_{tag}", "kind": kind, "layers": [], "size": 16, "animated": False,
            "license_verdict": "synthetic", "cf_license": "", "clip_rank": 0,
            "sha1": hashlib.sha1(sprite.tobytes()).hexdigest(),
        })
        per_concept[r["en"]] += 1

    since_save = 0
    for start in range(0, len(records), args.batch):
        if args.hours and time.time() - started > args.hours * 3600:
            print("time limit reached", flush=True)
            break
        # 几张底稿凑成一批一起画，显卡才吃得满
        group = records[start:start + args.batch]
        directs = [np.asarray(Image.open(args.src / "sprite" / r["file"]).convert("RGBA")) for r in group]
        for r, direct in zip(group, directs):
            add(r, direct, "d", "manual")
        inits = [as_canvas(direct, state) for direct in directs]
        for si, s in enumerate(strengths):
            gens = [torch.Generator(device=device).manual_seed(7000 + r["id"] * 10 + r["k"] * 3 + si + 100003 * j)
                    for r in group for j in range(per)]
            images = pipe(prompt=[state["trigger"] + r["en"] for r in group for _ in range(per)],
                          image=[init for init in inits for _ in range(per)], strength=s,
                          num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gens).images
            for idx, image in enumerate(images):
                r, direct, k = group[idx // per], directs[idx // per], idx % per
                sprite, _ = read_grid(image, state)
                problem = check(sprite)
                if problem is None and overlap(sprite, direct) < args.min_overlap:
                    problem = "drifted"
                if problem:
                    dropped[problem] += 1
                else:
                    add(r, sprite, f"r{si}{k}", "manual_redraw")
        n = start + len(group)
        rate = (time.time() - started) / n
        print(f"{n}/{len(records)} source images, {len(meta)} sprites so far, "
              f"about {(len(records) - n) * rate / 60:.0f} min left", flush=True)
        since_save += len(group)
        if since_save >= 48:
            save()  # 中途断掉也只丢最近几十张
            since_save = 0

    save()
    new = meta[first_new:]
    print(f"{summary(new) if new else 'nothing new'}; dropped redraws: {dict(dropped)}")
    print(f"dataset: {args.out}")


if __name__ == "__main__":
    main()
