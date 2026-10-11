"""把"训练方案对比实验"需要的文件打成一个 zip，传到租用的显卡上用。

Writes cloud/staritem_arms.zip: the training and evaluation code, the concept lists, the four training datasets
(only the 16x16 images of dataset/clean) and the cached CLIP features of the synthetic sprites. On the rented
machine, unpack it under /root and run cloud/arms.sh.

Usage:
  python -X utf8 tools/pack_arms_for_cloud.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["train/train.py", "train/model.py", "train/diffusion.py", "train/colors.py", "train/eval_bind.py", "train/eval_mod.py",
        "synth/build_dataset.py", "synth/pilot.py", "synth/generate.py", "synth/sprites.py",
        "cloud/arms.sh", "cloud/arms2.sh", "cloud/full.sh", "cloud/requirements.txt"]


def main():
    target = ROOT / "cloud" / "staritem_arms.zip"
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in CODE:
            z.write(ROOT / name, f"staritem/{name}")
        for path in sorted((ROOT / "synth" / "concepts").glob("*.tsv")):
            z.write(path, f"staritem/synth/concepts/{path.name}")
        for dataset in ("clean", "synth2", "many", "manual"):
            base = ROOT / "dataset" / dataset
            z.write(base / "metadata.jsonl", f"staritem/dataset/{dataset}/metadata.jsonl")
            for line in (base / "metadata.jsonl").open(encoding="utf-8"):
                r = json.loads(line)
                if r["size"] == 16:
                    z.write(base / r["file"], f"staritem/dataset/{dataset}/{r['file']}")
                    count += 1
        cache = ROOT / "runs" / "_bind_cache.npz"
        if cache.exists():
            z.write(cache, "staritem/runs/_bind_cache.npz", compress_type=zipfile.ZIP_STORED)
    print(f"{target} ({target.stat().st_size / 1e6:.0f} MB, {count} images)")


if __name__ == "__main__":
    main()
