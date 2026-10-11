"""用导出的 ONNX 模型按客户端的流程出图，把几个版本并排比较。

Each row is one prompt; each model gets three images. Name completion uses the model's own aliases.json.

Usage:
  .venv/Scripts/python.exe -X utf8 train/compare_exports.py --models SIC-1.1,SIC-1.2 --prompts "火鸡|拿铁|guitar"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import ort_sample  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLOAT_TEXT = None


def expand(text, aliases):
    """和客户端 engine.rs 的 lookup 一致：提示词以某个叫法结尾时补全成"中文名, 英文名"。"""
    def ends(key):
        if not text.endswith(key):
            return False
        before = text[:len(text) - len(key)]
        return not (key[0].isascii() and before and before[-1].isascii() and before[-1].isalnum())

    found = [k for k in aliases if ends(k)]
    if not found:
        return text, None
    key = max(found, key=len)
    pair, style = aliases[key]
    if key == text and text in pair.split(", "):
        return pair, style
    return (text if pair in text else f"{text}, {pair}"), style


class Model:
    def __init__(self, folder, style=None):
        self.style = style  # 给补全命中的物品强制指定画风，用来比较不同取值
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.aliases = json.loads((folder / "aliases.json").read_text(encoding="utf-8")) if (folder / "aliases.json").exists() else {}
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.unet = ort.InferenceSession(str(folder / "unet.onnx"), providers=providers)
        self.text = ort.InferenceSession(str(folder / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.max_len = self.manifest["text"]["max_len"]

    def encode(self, prompt):
        if FLOAT_TEXT is not None:
            # 用未量化的原始文本编码器，检查量化是否影响出图
            ctx, mask = FLOAT_TEXT.encode([prompt])
            return ctx.float().cpu().numpy(), mask.cpu().numpy()
        # 导出的 tokenizer 自带截断和补齐到 max_len，mask 要用它给的，不能自己按长度算
        enc = self.tokenizer.encode(prompt)
        ids = np.array([enc.ids], dtype=np.int64)
        att = np.array([enc.attention_mask], dtype=np.int64)
        feats = self.text.run(None, dict(zip([i.name for i in self.text.get_inputs()], [ids, att])))[0]
        return feats.astype(np.float32), att.astype(bool)

    def generate(self, prompt, count, seed):
        text, style = expand(prompt.strip().lower(), self.aliases)
        ctx, mask = self.encode(text)
        null_ctx, null_mask = self.encode("")
        d = self.manifest["defaults"]
        if style is not None and self.style is not None:
            style = self.style
        source = (style if style is not None else d.get("source", 0.0)) if self.manifest.get("has_source") else None
        x = ort_sample(self.unet, np.repeat(ctx, count, 0), np.repeat(mask, count, 0), null_ctx, null_mask, 16, 0,
                       d["steps"], d["guidance"], seed, source)
        rgba = np.clip((x.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).round().astype(np.uint8)
        rgba[..., 3] = np.where(rgba[..., 3] >= 128, 255, 0)
        return rgba, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="SIC-1.1,SIC-1.2",
                    help="folders under export/; NAME@0.7 forces that style mix for completed names")
    ap.add_argument("--prompts", required=True, help="prompts separated by |")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "export" / "compare.png")
    ap.add_argument("--float-text", action="store_true", help="use the original PyTorch text encoder instead of the exported one")
    args = ap.parse_args()
    if args.float_text:
        import torch
        from train import TextEncoder

        global FLOAT_TEXT
        FLOAT_TEXT = TextEncoder("intfloat/multilingual-e5-small", 32, torch.device("cuda"))

    names = args.models.split(",")
    models = [Model(ROOT / "export" / n.split("@")[0], float(n.split("@")[1]) if "@" in n else None) for n in names]
    prompts = [p for p in args.prompts.split("|") if p.strip()]
    cell, label, gap = 80, 170, 24
    block = args.count * cell + gap
    sheet = Image.new("RGB", (label + len(models) * block, 22 + len(prompts) * (cell + 6)), (58, 58, 68))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    for m, name in enumerate(names):
        draw.text((label + m * block, 2), name, fill=(255, 220, 120), font=font)
    for row, prompt in enumerate(prompts):
        y = 22 + row * (cell + 6)
        draw.text((4, y + 28), prompt[:20], fill=(235, 235, 235), font=font)
        for m, model in enumerate(models):
            sprites, text = model.generate(prompt, args.count, args.seed + row)
            for k, sprite in enumerate(sprites):
                im = Image.fromarray(sprite, "RGBA").resize((cell, cell), Image.NEAREST)
                sheet.paste(im, (label + m * block + k * cell, y), im)
            if m == len(models) - 1:
                print(f"{prompt} -> {text}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
