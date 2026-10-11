"""用训练好的 LoRA 给物品清单里的每个物品直接生成 16x16 贴图。可中断，重跑会接着做。

Reads synth/concepts/*.tsv, writes dataset/synth2_raw/: sprite/<id>_<k>.png and records.jsonl.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/generate_lora.py --hours 1.9
  .venv/Scripts/python.exe -X utf8 synth/generate_lora.py --names synth/concepts/pilot150.txt --per 20 --batch 1 --out dataset/many_raw
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import load_concepts  # noqa: E402
from lora_sprites import check, load_pipeline, read_grid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "synth2_raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=2, help="images per concept")
    ap.add_argument("--batch", type=int, default=2, help="concepts per pipeline call")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = run to the end)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", type=Path, help="text file with one English concept name per line; only these are generated")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--chunk", type=int, default=6, help="images per pipeline call")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--shard", default="", help="I/N: do only part I of N (one process per GPU); see run_shards.py")
    args = ap.parse_args()
    out = args.out

    concepts = load_concepts()
    random.Random(0).shuffle(concepts)  # 中途停下时各门类的覆盖是均匀的
    if args.names:
        wanted = set(args.names.read_text(encoding="utf-8").splitlines())
        concepts = [c for c in concepts if c["en"] in wanted]
    if args.limit:
        concepts = concepts[:args.limit]
    (out / "sprite").mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    # 分片时每个进程写自己的记录文件，互不干扰；已完成的物品要把所有记录文件都算上
    done = set()
    for path in out.glob("records*.jsonl"):
        done |= {json.loads(line)["id"] for line in path.open(encoding="utf-8")}
    if args.shard:
        part, parts = (int(v) for v in args.shard.split("/"))
        concepts = concepts[part::parts]
        records_path = out / f"records_shard{part}.jsonl"
    todo = [c for c in concepts if c["id"] not in done]
    print(f"concepts: {len(concepts)}, already done: {len(concepts) - len(todo)}, to do: {len(todo)}", flush=True)
    if not todo:
        return

    device = torch.device("cuda")
    pipe, state = load_pipeline(device)
    start = time.time()
    made = usable = 0
    with records_path.open("a", encoding="utf-8", newline="\n") as log:
        for i in range(0, len(todo), args.batch):
            if args.hours and time.time() - start > args.hours * 3600:
                print("time limit reached", flush=True)
                break
            group = todo[i:i + args.batch]
            jobs = [(c, k) for c in group for k in range(args.per)]
            lines = []
            for j in range(0, len(jobs), args.chunk):
                part = jobs[j:j + args.chunk]
                generator = torch.Generator(device=device).manual_seed(args.seed + group[0]["id"] * 100 + j)
                images = pipe(prompt=[state["trigger"] + c["en"] for c, _ in part], num_inference_steps=args.steps,
                              guidance_scale=args.guidance, height=state["res"], width=state["res"], generator=generator).images
                for (concept, k), image in zip(part, images):
                    sprite, spread = read_grid(image, state)
                    problem = check(sprite)
                    name = f"{concept['id']:05d}_{k}.png"
                    if problem is None:
                        Image.fromarray(sprite, "RGBA").save(out / "sprite" / name)
                        usable += 1
                    made += 1
                    lines.append(json.dumps({**concept, "k": k, "file": None if problem else name, "problem": problem,
                                             "spread": round(spread, 2)}, ensure_ascii=False) + "\n")
            # 一组物品全部出完才落盘，中途停下时不会留下只有一部分图的物品
            log.writelines(lines)
            log.flush()
            if (i // args.batch) % 10 == 0:
                elapsed = time.time() - start
                left = (len(todo) - i - len(group)) * args.per * elapsed / max(made, 1)
                print(f"{i + len(group):5}/{len(todo)} concepts, {usable}/{made} usable, "
                      f"{elapsed / max(made, 1):.2f} s/img, about {left / 60:.0f} min left", flush=True)
    print(f"done: {made} images, {usable} usable, {(time.time() - start) / 60:.0f} min")


if __name__ == "__main__":
    main()
