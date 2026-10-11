"""试跑：用 SDXL + 像素画 LoRA 给 50 个物品各生成 3 张，缩成 16x16，看缩小后还认不认得出来。

Needs the models from synth/download_models.py; this script never downloads anything.
Writes synth/pilot_out/: sheet.png (128px originals next to their 16px sprites), report.csv, sprites/.

Usage:
  .venv/Scripts/python.exe -X utf8 synth/pilot.py
  .venv/Scripts/python.exe -X utf8 synth/pilot.py --limit 10      # quicker look
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprites import remove_background, to_blocks, to_sprite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "synth" / "pilot_out"
PROMPT = "pixel, {thing}, single game item icon, isolated on plain white background, centered"
NEGATIVE = "3d render, realistic, photo, text, watermark, multiple objects, scenery, shadow, frame, border, card, paper, lines, plate, table"

# (英文提示词, 中文名)。最后几个是模型本来就会画的，用来对照
CONCEPTS = [
    ("a roast turkey", "火鸡"), ("a cup of latte coffee", "拿铁"), ("a slice of pizza", "披萨"), ("a hamburger", "汉堡"),
    ("a sushi roll", "寿司"), ("a donut", "甜甜圈"), ("an ice cream cone", "冰淇淋"), ("a hot dog", "热狗"),
    ("a croissant", "牛角包"), ("a bowl of ramen", "拉面"), ("a cup of bubble tea", "珍珠奶茶"), ("a birthday cake", "生日蛋糕"),
    ("an acoustic guitar", "吉他"), ("a violin", "小提琴"), ("a camera", "相机"), ("a smartphone", "手机"),
    ("a laptop computer", "笔记本电脑"), ("headphones", "耳机"), ("an umbrella", "雨伞"), ("a backpack", "背包"),
    ("a teapot", "茶壶"), ("a lantern", "灯笼"), ("a compass", "指南针"), ("an hourglass", "沙漏"),
    ("a candle", "蜡烛"), ("a magnifying glass", "放大镜"), ("a telescope", "望远镜"), ("an alarm clock", "闹钟"),
    ("a light bulb", "灯泡"), ("a battery", "电池"), ("a sneaker shoe", "运动鞋"), ("a top hat", "礼帽"),
    ("a pair of glasses", "眼镜"), ("a scarf", "围巾"), ("a goldfish", "金鱼"), ("a cactus in a pot", "仙人掌"),
    ("a seashell", "贝壳"), ("a rubber duck", "橡皮鸭"), ("a bicycle", "自行车"), ("a rocket", "火箭"),
    ("a basketball", "篮球"), ("a skateboard", "滑板"), ("a pair of dice", "骰子"), ("a gift box", "礼物盒"),
    ("a trophy cup", "奖杯"), ("a royal crown", "王冠"),
    ("an iron sword", "铁剑"), ("a red apple", "苹果"), ("a potion bottle", "药水瓶"), ("a wooden axe", "木斧"),
]

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def cached_dir(repo, filename):
    """返回缓存里某个仓库快照的目录。按目录加载，diffusers 就不会因为仓库里还有没下载的可选文件而报错。"""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename)
    if not isinstance(path, str):
        sys.exit(f"{repo}/{filename} is not in the cache; run synth/download_models.py first")
    return Path(path).parents[len(Path(filename).parts) - 1]


def load_pipeline(device):
    """模型由 download_models.py 提前下好，这里只读本地缓存，绝不联网下载。"""
    from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(cached_dir("madebyollin/sdxl-vae-fp16-fix", "config.json"), torch_dtype=torch.float16)
    base = cached_dir("stabilityai/stable-diffusion-xl-base-1.0", "model_index.json")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm, pixel = "pytorch_lora_weights.safetensors", "pixel-art-xl.safetensors"
    pipe.load_lora_weights(cached_dir("latent-consistency/lcm-lora-sdxl", lcm), weight_name=lcm, adapter_name="lcm")
    pipe.load_lora_weights(cached_dir("nerijs/pixel-art-xl", pixel), weight_name=pixel, adapter_name="pixel")
    pipe.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, 1.2])
    pipe.fuse_lora()  # 把 LoRA 并进权重，推理时少一层开销
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()  # 一张一张解码；整批解码会把 16 GB 显存撑爆，速度掉到几乎不动
    return pipe.to(device)


def on_gray(rgba, size):
    """RGBA 贴图 -> 灰底上的 RGB，最近邻放大到 size，供 CLIP 看。"""
    a = rgba[..., 3:].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 128.0 * (1 - a)
    img = Image.fromarray(rgb.round().astype(np.uint8)).resize((size, size), Image.NEAREST)
    return np.asarray(img)


class Clip:
    """用 CLIP 做"这张图像哪个物品"的选择题，衡量缩小后还认不认得出。"""

    def __init__(self, device, texts):
        from transformers import AutoTokenizer, CLIPModel

        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device).eval()
        tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        enc = tok(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            self.text = self._norm(self.model.get_text_features(**enc))

    @staticmethod
    def _norm(out):
        feats = out if torch.is_tensor(out) else out.pooler_output
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def rank(self, images, answers):
        """images: (N,224,224,3) uint8。返回每张图的正确答案排第几（1 = 第一）。"""
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - CLIP_MEAN) / CLIP_STD).to(self.device)
        sims = self._norm(self.model.get_image_features(pixel_values=x)) @ self.text.T
        order = sims.argsort(dim=-1, descending=True)
        return [int((order[i] == answers[i]).nonzero()[0, 0]) + 1 for i in range(len(images))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(CONCEPTS), help="only the first N concepts")
    ap.add_argument("--per", type=int, default=3, help="images per concept")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--res", type=int, default=768, help="generation resolution; the pixel blocks are 8px, so 768 gives 96x96 art")
    ap.add_argument("--prompt", default=PROMPT, help="prompt template with {thing}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    concepts = CONCEPTS[:args.limit]
    (OUT / "sprites").mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    records, big, small = [], [], []
    seconds = []
    for ci, (thing, zh) in enumerate(concepts):
        t0 = time.time()
        images = pipe(prompt=[args.prompt.format(thing=thing)] * args.per, negative_prompt=[NEGATIVE] * args.per,
                      num_inference_steps=args.steps, guidance_scale=args.guidance, height=args.res, width=args.res,
                      generator=generator).images
        if ci > 0:  # 第一批包含预热，不计入速度
            seconds.append((time.time() - t0) / args.per)
        for k, image in enumerate(images):
            blocks = to_blocks(image)
            mask = remove_background(blocks)
            sprite, problem = to_sprite(blocks, mask)
            cut = np.dstack([blocks, np.where(mask, 255, 0).astype(np.uint8)])
            big.append(cut)
            small.append(sprite)
            if sprite is not None:
                Image.fromarray(sprite, "RGBA").save(OUT / "sprites" / f"{ci:02d}_{k}.png")
            records.append({"concept": thing, "zh": zh, "index": k, "problem": problem or "", "answer": ci})
        print(f"[{ci + 1:2}/{len(concepts)}] {zh} {thing}: " + " ".join(r["problem"] or "ok" for r in records[-args.per:]), flush=True)

    del pipe
    torch.cuda.empty_cache()

    # CLIP 选择题：在全部物品里选，正确答案排第几
    clip = Clip(device, [f"a pixel art icon of {thing}" for thing, _ in concepts])
    answers = [r["answer"] for r in records]
    rank_big = clip.rank([np.asarray(Image.fromarray(on_gray(b, 128)).resize((224, 224), Image.BICUBIC)) for b in big], answers)
    usable = [i for i, s in enumerate(small) if s is not None]
    rank_small = dict(zip(usable, clip.rank([on_gray(small[i], 224) for i in usable], [answers[i] for i in usable]))) if usable else {}
    for i, r in enumerate(records):
        r["rank_128"] = rank_big[i]
        r["rank_16"] = rank_small.get(i, "")
        del r["answer"]
    with (OUT / "report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    # 对照图：每个物品一行，左边是 128px 原图，右边是 16px 贴图放大
    cell, pad = 128, 6
    cols = args.per * 2
    sheet = Image.new("RGBA", (cols * (cell + pad) + pad + 14, len(concepts) * (cell + pad) + pad), (58, 58, 68, 255))
    for i, (b, s) in enumerate(zip(big, small)):
        row, k = divmod(i, args.per)
        y = pad + row * (cell + pad)
        sheet.paste(Image.fromarray(on_gray(b, cell)), (pad + k * (cell + pad), y))  # 128 以外的分辨率会被缩放到格子大小
        x = 14 + pad + (args.per + k) * (cell + pad)
        if s is not None:
            sheet.paste(Image.fromarray(on_gray(s, cell)), (x, y))
    sheet.save(OUT / "sheet.png")

    n = len(records)
    top = lambda ranks, k: sum(1 for r in ranks if r != "" and r <= k)  # noqa: E731
    r16 = [r["rank_16"] for r in records]
    problems = {}
    for r in records:
        if r["problem"]:
            problems[r["problem"]] = problems.get(r["problem"], 0) + 1
    print(f"\nimages: {n}, usable sprites: {len(usable)} ({len(usable) / n:.0%}), dropped: {problems}")
    print(f"CLIP picks the right concept out of {len(concepts)}:")
    print(f"  128px originals: top-1 {top(rank_big, 1) / n:.0%}, top-5 {top(rank_big, 5) / n:.0%}")
    if usable:
        print(f"  16px sprites:    top-1 {top(r16, 1) / len(usable):.0%}, top-5 {top(r16, 5) / len(usable):.0%} (of usable)")
    if seconds:
        print(f"speed: {sum(seconds) / len(seconds):.2f} s per image")
    print(f"sheet: {OUT / 'sheet.png'}")


if __name__ == "__main__":
    main()
