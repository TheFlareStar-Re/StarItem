"""训练物品贴图扩散模型。

Reads dataset/clean (metadata.jsonl + images), writes runs/<name>/:
  config.json, log.csv, ckpt_last.pt (full state, for --resume), ema_<step>.pt (EMA weights), samples/<step>.png

Usage:
  python -X utf8 train/train.py                       # 16x16 only, English names
  python -X utf8 train/train.py --resume              # continue runs/<name>
  python -X utf8 train/train.py --sizes 16,32 --name dual
  python -X utf8 train/train.py --smoke               # 20 steps with a random stand-in text encoder
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colors import dominant_color, with_color  # noqa: E402
from diffusion import T, cosine_alphas_cumprod, ddim_sample, min_snr_weight, q_sample  # noqa: E402
from model import UNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIZE_INDEX = {16: 0, 32: 1}
PREVIEW_EN = ["diamond sword", "iron pickaxe", "golden apple", "red potion", "emerald ring", "cooked fish",
              "copper ingot", "magic wand", "leather boots", "blue gem", "wooden shield", "bowl of soup"]
PREVIEW_ZH = ["钻石剑", "铁镐", "金苹果", "红色药水", "绿宝石戒指", "熟鱼", "铜锭", "魔杖", "皮革靴子", "蓝宝石", "木盾", "一碗汤"]
# 训练集加入合成数据后，预览里换上几个模组贴图里没有的物品
PREVIEW_MIXED = ["diamond sword", "red potion", "roast turkey", "acoustic guitar", "smartphone", "blue umbrella",
                 "钻石剑", "铜锭", "感恩节火鸡", "拿铁", "寿司", "红色的自行车"]
STOPWORDS = {"of", "the", "a", "an", "and", "with", "in", "on", "to", "for"}
CJK = re.compile(r"[一-鿿]")


class TextEncoder:
    """冻结的文本编码器，输出 token 级特征和 padding mask。"""

    def __init__(self, name, max_len, device):
        from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel
        self.max_len = max_len
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        # CLIP 只加载文本塔，不要图像塔
        cls = CLIPTextModel if AutoConfig.from_pretrained(name).model_type == "clip" else AutoModel
        self.model = cls.from_pretrained(name).to(device).eval().requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, batch=256):
        outs, masks = [], []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding="max_length", truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.device)
            outs.append(self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state)
            masks.append(enc.attention_mask.bool())
        return torch.cat(outs).float(), torch.cat(masks)


class TokenMap:
    """分词器的词表 -> 紧凑编号，只收训练提示词里出现过的词。0 号留给没见过的词、特殊符号和补齐位。"""

    def __init__(self, ids):
        self.ids = list(ids)
        self.index = {t: i + 1 for i, t in enumerate(self.ids)}

    @staticmethod
    def raw(tok, texts, max_len):
        enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        special = torch.tensor(sorted(tok.all_special_ids))
        return torch.where(torch.isin(enc.input_ids, special), torch.full_like(enc.input_ids, -1), enc.input_ids)

    @classmethod
    def build(cls, tok, texts, max_len):
        ids = torch.cat([cls.raw(tok, texts[i:i + 4096], max_len).flatten().unique() for i in range(0, len(texts), 4096)])
        return cls(sorted(set(ids.tolist()) - {-1}))

    def encode(self, tok, texts, max_len, device):
        table = torch.zeros(len(tok) + 1, dtype=torch.long)  # 最后一格接住 -1（特殊符号）
        table[torch.tensor(self.ids)] = torch.arange(1, len(self.ids) + 1)
        table[-1] = 0
        return torch.cat([table[self.raw(tok, texts[i:i + 4096], max_len)] for i in range(0, len(texts), 4096)]).to(device)

    def save(self, path):
        path.write_text(json.dumps({"ids": self.ids}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8"))["ids"])


class DummyTextEncoder:
    """--smoke 用：按字符串哈希出固定的随机特征，不需要下载任何模型。"""

    dim = 64

    def __init__(self, max_len, device):
        self.max_len = max_len
        self.device = device

    def encode(self, texts, batch=256):
        out = torch.zeros(len(texts), self.max_len, self.dim)
        mask = torch.zeros(len(texts), self.max_len, dtype=torch.bool)
        for i, s in enumerate(texts):
            n = min(self.max_len, 1 + len(s.split()))
            g = torch.Generator().manual_seed(sum(s.encode("utf-8")) + 31 * len(s))
            out[i, :n] = torch.randn(n, self.dim, generator=g)
            mask[i, :n] = True
        return out.to(self.device), mask.to(self.device)


def en_variants(name):
    """英文名的截短写法：只留中心词、只留修饰词等。"diamond sword" -> "sword", "diamond"。"""
    w = name.split()
    if len(w) < 2:
        return []
    cands = {w[-1], " ".join(w[:-1])}
    if len(w) >= 3:
        cands |= {" ".join(w[-2:]), " ".join(w[1:])}

    def meaningful(s):
        ws = [x for x in s.split() if x not in STOPWORDS and not re.fullmatch(r"[0-9]+|[ivx]+", x)]
        return sum(len(x) for x in ws) >= 3

    return [c for c in cands if c != name and meaningful(c)]


def zh_affixes(names, min_count=10):
    """中文名没有空格，用"在很多名称里出现的前后缀"当作类别词，如"剑""锭""钻石"。"""
    count = Counter()
    for n in set(names):
        for k in (1, 2, 3):
            if len(n) > k:
                count["s", n[-k:]] += 1
        for k in (2, 3):
            if len(n) > k:
                count["p", n[:k]] += 1
    keep = {k for k, v in count.items() if v >= min_count}
    # "示牌"几乎只出现在"告示牌"里，这种被更长词缀完全解释的片段不是词，去掉
    for kind, s in list(keep):
        for (k2, t), v in count.items():
            if k2 == kind and len(t) > len(s) and (t.endswith(s) if kind == "s" else t.startswith(s)) \
                    and v >= 0.9 * count[kind, s]:
                keep.discard((kind, s))
                break
    return keep


def zh_variants(name, affixes):
    out = [name[-k:] for k in (1, 2, 3) if len(name) > k and ("s", name[-k:]) in affixes]
    out += [name[:k] for k in (2, 3) if len(name) > k and ("p", name[:k]) in affixes]
    return [v for v in out if v.strip() == v and v]


def load_dataset(args, device):
    """返回按分辨率分桶的图像张量，以及每张图可用的提示词。"""
    rows = []
    for source, folder in enumerate([args.data, *args.extra]):
        part = [json.loads(line) for line in (folder / "metadata.jsonl").open(encoding="utf-8")]
        for r in part:
            r["path"] = folder / r["file"]
            r["source"] = min(source, 1)  # 0 = 手绘的模组贴图，1 = 合成图
        rows += part[:512] if args.smoke else part
    rows = [r for r in rows if r["size"] in args.sizes and not (args.no_nc and r["license_verdict"] == "ok_nc")]
    captions = {"": 0}  # 0 号是空提示词，用于 classifier-free guidance

    def names_of(r):
        names = []
        if args.lang in ("en", "both"):
            names += [r["name_en"], *r["alt_names_en"]]
        if args.lang in ("zh", "both") and r["name_zh"]:
            names += [r["name_zh"], *r.get("alt_names_zh", [])]
        return [n.lower() for n in (names or [r["name_en"]])]

    def extras_of(r, image):
        """补充写法：带颜色词的名称，以及"别名, 中文名, 英文名"这种客户端补全后的形式。"""
        out = []
        color = dominant_color(image, 0.6)
        if color:
            if args.lang in ("en", "both"):
                out.append(with_color(r["name_en"].lower(), color, False))
            if args.lang in ("zh", "both") and r["name_zh"]:
                out.append(with_color(r["name_zh"].lower(), color, True))
        if args.lang == "both" and r["name_zh"]:
            pair = f"{r['name_zh']}, {r['name_en']}".lower()
            out.append(pair)
            out += [f"{a}, {pair}".lower() for a in [*r["alt_names_en"], *r.get("alt_names_zh", [])][:4]]
        return [c for c in out if c]

    affixes = zh_affixes([n for r in rows for n in names_of(r) if CJK.search(n)])

    def pad(lists):
        width = max(1, max(len(c) for c in lists))
        ids = torch.tensor([c + [c[0] if c else 0] * (width - len(c)) for c in lists], device=device)
        return ids, torch.tensor([len(c) for c in lists], device=device)

    buckets = {}
    for size in args.sizes:
        sub = [r for r in rows if r["size"] == size]
        if not sub:
            continue
        imgs = np.stack([np.asarray(Image.open(r["path"]).convert("RGBA")) for r in sub])
        full, aug, extra = [], [], []
        for r, image in zip(sub, imgs):
            names = names_of(r)
            short = sorted({v for n in names for v in (zh_variants(n, affixes) if CJK.search(n) else en_variants(n))})
            full.append([captions.setdefault(n, len(captions)) for n in names])
            aug.append([captions.setdefault(n, len(captions)) for n in short])
            extra.append([captions.setdefault(n, len(captions)) for n in extras_of(r, image)])
        cap_ids, cap_n = pad(full)
        aug_ids, aug_n = pad(aug)
        extra_ids, extra_n = pad(extra)
        buckets[size] = {
            "images": torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device),  # uint8 (N,4,S,S)
            "cap_ids": cap_ids, "cap_n": cap_n, "aug_ids": aug_ids, "aug_n": aug_n,
            "extra_ids": extra_ids, "extra_n": extra_n,
            "source": torch.tensor([r["source"] for r in sub], device=device),
        }
    return buckets, list(captions)


def to_rgba_uint8(x):
    """模型输出 [-1,1] -> uint8 RGBA，alpha 二值化，透明像素清零。"""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy().copy()
    opaque = x[..., 3] >= 128
    x[..., 3] = np.where(opaque, 255, 0)
    x[~opaque] = 0
    return x


def save_grid(images, cols, path, scale=6, bg=(58, 58, 68, 255)):
    n, s = images.shape[0], images.shape[1]
    rows = math.ceil(n / cols)
    cell = s * scale + 4
    sheet = Image.new("RGBA", (cols * cell + 4, rows * cell + 4), bg)
    for i, im in enumerate(images):
        tile = Image.fromarray(im, "RGBA").resize((s * scale, s * scale), Image.NEAREST)
        sheet.alpha_composite(tile, (4 + (i % cols) * cell, 4 + (i // cols) * cell))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def main():
    ap = argparse.ArgumentParser(description="Train the item texture diffusion model.")
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "clean")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="additional datasets of synthetic sprites")
    ap.add_argument("--merge-extra", action="store_true",
                    help="treat --extra sprites as the same style as --data (the model gets no style input)")
    ap.add_argument("--name", default="bi16", help="run name under runs/")
    ap.add_argument("--init", help="start from the weights of another run (its ckpt_last.pt); layers it lacks start fresh")
    ap.add_argument("--sizes", default="16", help="resolutions to train on: 16 or 16,32")
    ap.add_argument("--lang", choices=["en", "zh", "both"], default="both", help="which names to use as prompts")
    ap.add_argument("--text-encoder", default="intfloat/multilingual-e5-small",
                    help="frozen text encoder; openai/clip-vit-base-patch32 is English only")
    ap.add_argument("--max-len", type=int, default=32, help="prompt length in tokens")
    ap.add_argument("--cap-aug", type=float, default=0.3, help="probability of training on a shortened name")
    ap.add_argument("--cap-aug-synth", type=float, help="the same for --extra sprites; default: same as --cap-aug")
    ap.add_argument("--cap-extra", type=float, default=0.2, help="probability of a color-tagged or expanded name")
    ap.add_argument("--tok-emb", action="store_true",
                    help="add a trainable vector per token to the frozen text features")
    ap.add_argument("--tok-lr-mult", type=float, default=5.0, help="learning rate of those vectors, as a multiple of --lr")
    ap.add_argument("--tok-drop", type=float, default=0.0,
                    help="probability of training a sample without its token vectors, so the model can still work from the frozen features")
    ap.add_argument("--no-nc", action="store_true", help="exclude images under non-commercial licenses")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--cond-drop", type=float, default=0.1, help="probability of dropping the prompt (for CFG)")
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--guidance", type=float, default=3.0, help="CFG scale for preview samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0, help="save and exit after this long; continue with --resume")
    ap.add_argument("--smoke", action="store_true", help="quick self-test: tiny subset, 20 steps, dummy text encoder")
    args = ap.parse_args()
    args.sizes = [int(s) for s in str(args.sizes).split(",")]
    assert all(s in SIZE_INDEX for s in args.sizes), "sizes must be 16 and/or 32"
    if args.smoke:
        args.name, args.steps, args.save_every, args.batch, args.warmup = "smoke", 20, 10, 32, 5

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run = ROOT / "runs" / args.name
    if (run / "ckpt_last.pt").exists() and not args.resume and not args.smoke:
        sys.exit(f"{run} already has a checkpoint. Use --resume to continue it or --name for a new run.")
    run.mkdir(parents=True, exist_ok=True)
    preview_prompts = {"en": PREVIEW_EN, "zh": PREVIEW_ZH, "both": PREVIEW_EN[:6] + PREVIEW_ZH[:6]}[args.lang]
    if args.extra and args.lang == "both":
        preview_prompts = PREVIEW_MIXED

    buckets, captions = load_dataset(args, device)
    if not buckets:
        sys.exit("No images match --sizes.")
    counts = {s: b["images"].shape[0] for s, b in buckets.items()}
    print(f"device={device} images={counts} prompts={len(captions)}", flush=True)

    text = DummyTextEncoder(args.max_len, device) if args.smoke else TextEncoder(args.text_encoder, args.max_len, device)
    t0 = time.time()
    emb, emb_mask = text.encode(captions)
    emb = emb.half() if device.type == "cuda" else emb
    preview_emb, preview_mask = text.encode(preview_prompts)
    ctx_dim = text.dim
    tok_ids = preview_tok = None
    num_tokens = 0
    if args.tok_emb:
        assert not args.smoke, "--tok-emb needs the real tokenizer"
        map_path = run / "tokens.json"
        token_map = TokenMap.load(map_path) if args.resume and map_path.exists() else TokenMap.build(text.tok, captions, args.max_len)
        token_map.save(map_path)
        tok_ids = token_map.encode(text.tok, captions, args.max_len, device)
        preview_tok = token_map.encode(text.tok, preview_prompts, args.max_len, device)
        num_tokens = len(token_map.ids) + 1
        print(f"token vectors: {num_tokens - 1} tokens", flush=True)
    print(f"encoded prompts in {time.time() - t0:.1f}s, ctx_dim={ctx_dim}", flush=True)
    if not args.smoke:
        del text.model  # 特征已经全部算好，训练时不再需要编码器
        torch.cuda.empty_cache()

    cfg = {"ctx_dim": ctx_dim, "ch": args.ch, "mult": [1, 2, 3], "num_res": 2, "attn_levels": [1, 2], "heads": 4,
           "dropout": 0.1, "num_sizes": len(SIZE_INDEX), "num_sources": 2 if args.extra and not args.merge_extra else 0, "num_tokens": num_tokens}
    model = UNet(**cfg).to(device)
    ema = UNet(**cfg).to(device).eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("tok_emb.")], "mult": 1.0}]
    if num_tokens:
        # 词向量从零学起，每个词见到的次数又少：学习率放大，不做权重衰减
        groups.append({"params": list(model.tok_emb.parameters()), "mult": args.tok_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ac = cosine_alphas_cumprod().to(device)
    step = 0
    ckpt_path = run / "ckpt_last.pt"
    if args.init and not args.resume:
        ck = torch.load(ROOT / "runs" / args.init / "ckpt_last.pt", map_location=device, weights_only=True)
        if model.src_emb is not None:
            torch.nn.init.zeros_(model.src_emb.weight)  # 画风标记从零开始，起点和原模型完全一样
        missing, unexpected = model.load_state_dict(ck["ema"], strict=False)
        ema.load_state_dict(model.state_dict())
        print(f"initialized from {args.init} (step {ck['step']}); new layers: {missing}, unused: {unexpected}", flush=True)
        del ck
    if args.resume:
        if not ckpt_path.exists():
            sys.exit(f"--resume: {ckpt_path} not found")
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from step {step}", flush=True)
    (run / "config.json").write_text(json.dumps({
        "model": cfg, "text_encoder": "dummy" if args.smoke else args.text_encoder, "max_len": args.max_len,
        "lang": args.lang, "sizes": args.sizes, "prediction": "v", "schedule": "cosine", "timesteps": T,
        "images": counts, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    sizes = list(buckets)
    probs = torch.tensor([counts[s] for s in sizes], dtype=torch.float)
    probs /= probs.sum()
    log_new = not (run / "log.csv").exists() or not args.resume
    log = (run / "log.csv").open("w" if log_new else "a", encoding="utf-8", newline="")
    writer = csv.writer(log)
    if log_new:
        writer.writerow(["step", "loss", "lr", "it_per_s"])

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    def preview(s):
        g = torch.Generator(device=device).manual_seed(1234)
        for size in sizes:
            reps = 3  # 每个提示词出 3 张，和客户端的用法一致
            ctx = preview_emb.repeat_interleave(reps, 0)
            mask = preview_mask.repeat_interleave(reps, 0)
            with autocast(device):
                # 预览都按"手绘模组贴图"的风格出，这也是客户端要用的设置
                x = ddim_sample(ema, ctx.to(emb.dtype), mask, emb[:1], emb_mask[:1], size, SIZE_INDEX[size], ac,
                                steps=50, guidance=args.guidance, generator=g, source=0,
                                tok=preview_tok.repeat_interleave(reps, 0) if num_tokens else None,
                                null_tok=tok_ids[:1] if num_tokens else None)
            save_grid(to_rgba_uint8(x), reps * 4, run / "samples" / f"{s:06d}_{size}px.png", scale=6 if size == 16 else 3)

    def save(s):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": s}, ckpt_path)
        torch.save(ema.state_dict(), run / f"ema_{s:06d}.pt")

    model.train()
    running, t_last, s_last = None, time.time(), step
    started = time.time()
    while step < args.steps:
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            save(step)
            print(f"time limit reached at step {step}, checkpoint saved; continue with --resume", flush=True)
            break
        size = sizes[torch.multinomial(probs, 1).item()]
        b = buckets[size]
        idx = torch.randint(0, counts[size], (args.batch,), device=device)
        x0 = b["images"][idx].float() / 127.5 - 1.0
        pick = (torch.rand(args.batch, device=device) * b["cap_n"][idx]).long()
        cid = b["cap_ids"][idx, pick]
        # 一部分样本改用截短的名称，让"剑"这类词对应到很多张不同的图
        pick = (torch.rand(args.batch, device=device) * b["aug_n"][idx]).long()
        p_aug = args.cap_aug
        if args.cap_aug_synth is not None:
            p_aug = args.cap_aug + (args.cap_aug_synth - args.cap_aug) * (b["source"][idx] > 0).float()
        use_aug = (torch.rand(args.batch, device=device) < p_aug) & (b["aug_n"][idx] > 0)
        cid = torch.where(use_aug, b["aug_ids"][idx, pick], cid)
        pick = (torch.rand(args.batch, device=device) * b["extra_n"][idx]).long()
        use_extra = (torch.rand(args.batch, device=device) < args.cap_extra) & (b["extra_n"][idx] > 0)
        cid = torch.where(use_extra, b["extra_ids"][idx, pick], cid)
        cid = torch.where(torch.rand(args.batch, device=device) < args.cond_drop, torch.zeros_like(cid), cid)
        t = torch.randint(0, T, (args.batch,), device=device)
        xt, target = q_sample(x0, t, torch.randn_like(x0), ac)
        sidx = torch.full((args.batch,), SIZE_INDEX[size], device=device, dtype=torch.long)

        for group in opt.param_groups:
            group["lr"] = lr_at(step) * group.get("mult", 1.0)
        with autocast(device):
            tok = tok_ids[cid] if num_tokens else None
            if num_tokens and args.tok_drop > 0:
                # 这些样本的词编号全部换成 0 号（向量为零），只靠冻结的文本特征
                tok = tok * (torch.rand(args.batch, 1, device=device) >= args.tok_drop)
            pred = model(xt, t, emb[cid], emb_mask[cid], sidx, b["source"][idx], tok)
        loss = ((pred.float() - target) ** 2).mean(dim=(1, 2, 3))
        loss = (loss * min_snr_weight(t, ac)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        decay = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)

        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if not math.isfinite(running):
            sys.exit(f"loss became non-finite at step {step}")
        if step % 100 == 0 or step == args.steps:
            now = time.time()
            speed = (step - s_last) / max(1e-6, now - t_last)
            t_last, s_last = now, step
            writer.writerow([step, f"{running:.5f}", f"{lr_at(step):.2e}", f"{speed:.1f}"])
            log.flush()
            eta = (args.steps - step) / max(speed, 1e-6) / 60
            print(f"step {step:6d}/{args.steps} loss {running:.4f} {speed:.1f} it/s eta {eta:.0f} min", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(step)
            preview(step)
            print(f"saved checkpoint and samples at step {step}", flush=True)
    log.close()
    print(f"{'done' if step >= args.steps else 'paused'}: {run}")


if __name__ == "__main__":
    main()
