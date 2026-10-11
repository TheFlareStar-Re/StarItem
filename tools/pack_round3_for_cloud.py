"""把"第三批底稿重绘 + 下一版训练"要用的文件打成一个 zip，传到租用的机器上。

Writes cloud/staritem_round3.zip: the redraw and training code, our LoRA, the item lists, the finished datasets
(clean, synth2, many, manual) and the 16x16 drafts converted by import_manual.py. The SDXL base model is
downloaded on the rented machine.

Usage:
  python -X utf8 tools/pack_round3_for_cloud.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["runs/lora16/pytorch_lora_weights.safetensors", "runs/lora16/state.json", "cloud/requirements.txt", "cloud/round3.sh"]


def main():
    files = FILES + [p.relative_to(ROOT).as_posix() for folder in ("synth", "train") for p in sorted((ROOT / folder).glob("*.py"))]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "synth" / "concepts").iterdir()) if p.is_file()]
    target = ROOT / "cloud" / "staritem_round3.zip"
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in files:
            z.write(ROOT / name, f"staritem/{name}")
        for dataset in ("clean", "synth2", "many", "manual"):
            base = ROOT / "dataset" / dataset
            z.write(base / "metadata.jsonl", f"staritem/dataset/{dataset}/metadata.jsonl")
            for line in (base / "metadata.jsonl").open(encoding="utf-8"):
                rel = json.loads(line)["file"]
                z.write(base / rel, f"staritem/dataset/{dataset}/{rel}")
                count += 1
        # 底稿：重绘只做 dataset/manual 里还没有的，但记录文件要整份带上
        drafts = ROOT / "dataset" / "manual_raw16"
        z.write(drafts / "records.jsonl", "staritem/dataset/manual_raw16/records.jsonl")
        for line in (drafts / "records.jsonl").open(encoding="utf-8"):
            name = json.loads(line)["file"]
            if name:
                z.write(drafts / "sprite" / name, f"staritem/dataset/manual_raw16/sprite/{name}")
                count += 1
    print(f"{target} ({target.stat().st_size / 1e6:.0f} MB, {count} images, {len(files)} other files)")


if __name__ == "__main__":
    main()
