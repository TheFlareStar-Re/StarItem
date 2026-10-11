"""把在租用的显卡上跑"本地多图生成"需要的文件打成一个 zip。

Writes cloud/staritem_cloud.zip (about 190 MB): the generation scripts, the item lists, our LoRA and the
record of what is already done. The SDXL base model is not included; it is downloaded on the rented machine.

Usage:
  python -X utf8 tools/pack_for_cloud.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "synth/generate_lora.py", "synth/lora_sprites.py", "synth/generate.py", "synth/pilot.py", "synth/sprites.py",
    "synth/download_models.py", "synth/run_shards.py", "synth/concepts/rest.txt",
    "runs/lora16/pytorch_lora_weights.safetensors", "runs/lora16/state.json",
    "dataset/many_raw/records.jsonl", "cloud/README.md", "cloud/requirements.txt",
]


def main():
    files = FILES + [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "synth" / "concepts").glob("*.tsv"))]
    target = ROOT / "cloud" / "staritem_cloud.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(ROOT / name, f"staritem/{name}")
    print(f"{target} ({target.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
