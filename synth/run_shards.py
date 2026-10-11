"""在多张显卡上并行跑 generate_lora.py，以及查看进度、打包结果、合并记录。

Usage (on the rented machine, from the project folder):
  python -X utf8 synth/run_shards.py --gpus 4          # start one process per GPU, wait for all of them
  python -X utf8 synth/run_shards.py --status          # progress so far
  python -X utf8 synth/run_shards.py --pack            # write many_raw_cloud.tar to download
Usage (back on the local machine, after extracting the tar into the project folder):
  python -X utf8 synth/run_shards.py --merge
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "many_raw"


def status(names):
    wanted = set(names.read_text(encoding="utf-8").splitlines())
    done, images, usable = set(), 0, 0
    for path in OUT.glob("records*.jsonl"):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["en"] in wanted:
                done.add(r["en"])
                images += 1
                usable += r["file"] is not None
    print(f"{len(done)} of {len(wanted)} items done, {images} images, {usable} usable")
    return len(done) == len(wanted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=0, help="number of GPUs to use")
    ap.add_argument("--names", type=Path, default=ROOT / "synth" / "concepts" / "rest.txt")
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=10, help="images per pipeline call; 10 fits in 24 GB, use 5 on 16 GB")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.status:
        status(args.names)
    elif args.pack:
        target = ROOT / "many_raw_cloud.tar"
        with tarfile.open(target, "w") as tar:
            tar.add(OUT / "sprite", arcname="dataset/many_raw/sprite")
            for path in OUT.glob("records_shard*.jsonl"):
                tar.add(path, arcname=f"dataset/many_raw/{path.name}")
        print(f"{target} ({target.stat().st_size / 1e6:.0f} MB)")
    elif args.merge:
        # 把各分片的记录并进 records.jsonl，已有的物品不重复
        main_path = OUT / "records.jsonl"
        have = {json.loads(line)["id"] for line in main_path.open(encoding="utf-8")} if main_path.exists() else set()
        added = 0
        with main_path.open("a", encoding="utf-8", newline="\n") as f:
            for path in sorted(OUT.glob("records_shard*.jsonl")):
                for line in path.open(encoding="utf-8"):
                    if json.loads(line)["id"] not in have:
                        f.write(line if line.endswith("\n") else line + "\n")
                        added += 1
                path.unlink()
        print(f"merged {added} records into {main_path}")
    elif args.gpus:
        procs = []
        for i in range(args.gpus):
            log = (ROOT / f"gen_{i}.log").open("w", encoding="utf-8")
            cmd = [sys.executable, "-X", "utf8", str(ROOT / "synth" / "generate_lora.py"), "--names", str(args.names),
                   "--per", str(args.per), "--batch", "1", "--chunk", str(args.chunk), "--out", str(OUT), "--shard", f"{i}/{args.gpus}"]
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}))
        print(f"started {len(procs)} processes; logs are gen_0.log … gen_{args.gpus - 1}.log; "
              "run this script with --status in another terminal to see progress", flush=True)
        codes = [p.wait() for p in procs]
        print("exit codes:", codes)
        status(args.names)
    else:
        ap.error("give --gpus N, --status, --pack or --merge")


if __name__ == "__main__":
    main()
