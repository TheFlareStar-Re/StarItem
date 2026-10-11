"""把在租用的显卡上训练需要的文件打成一个 zip（接在 pack_for_cloud.py 的生成包之后用）。

Writes cloud/staritem_train.zip: the training code, the dataset builder, the three finished datasets
(clean, synth2, manual) and the many-image sprites generated locally so far. dataset/many itself is built
on the rented machine after generation finishes there.

Usage:
  python -X utf8 tools/pack_train_for_cloud.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["train/train.py", "train/model.py", "train/diffusion.py", "train/colors.py", "synth/build_dataset.py", "synth/run_shards.py"]


def main():
    target = ROOT / "cloud" / "staritem_train.zip"
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in CODE:
            z.write(ROOT / name, f"staritem/{name}")
        for dataset in ("clean", "synth2", "manual"):
            base = ROOT / "dataset" / dataset
            z.write(base / "metadata.jsonl", f"staritem/dataset/{dataset}/metadata.jsonl")
            for line in (base / "metadata.jsonl").open(encoding="utf-8"):
                rel = json.loads(line)["file"]
                z.write(base / rel, f"staritem/dataset/{dataset}/{rel}")
                count += 1
        # 本机已经生成的多图贴图：记录单独起名，免得盖掉云端包里那份
        raw = ROOT / "dataset" / "many_raw"
        z.write(raw / "records.jsonl", "staritem/dataset/many_raw/records_local.jsonl")
        for line in (raw / "records.jsonl").open(encoding="utf-8"):
            name = json.loads(line)["file"]
            if name:
                z.write(raw / "sprite" / name, f"staritem/dataset/many_raw/sprite/{name}")
                count += 1
    print(f"{target} ({target.stat().st_size / 1e6:.0f} MB, {count} images)")


if __name__ == "__main__":
    main()
