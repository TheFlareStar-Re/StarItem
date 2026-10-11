"""名字和物品对得上吗：模型按名字画出来的图，像不像这个物品自己的训练图。

CLIP has trouble connecting a 16x16 sprite to a written name, so the name-based rank in eval_recall.py is noisy:
plainly correct sprites can rank in the thousands. Here the yardstick is the training sprites themselves. Every
concept gets the average CLIP image feature of its synthetic training sprites, and a generated sprite is ranked by
how close it is to its own concept's average compared with all the other concepts'. The same measure applied to the
training sprites (each one against the averages, its own computed without it) gives the ceiling.

Usage:
  .venv/Scripts/python.exe -X utf8 train/eval_bind.py --run sic16 --step 150000 --per-concept export/dev/sic16_bind.csv
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "synth"))
from build_dataset import direct_ranks  # noqa: E402
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from generate import load_concepts  # noqa: E402
from model import UNet  # noqa: E402
from pilot import CLIP_MEAN, CLIP_STD, Clip  # noqa: E402
from train import TextEncoder, TokenMap, autocast, to_rgba_uint8  # noqa: E402


@torch.no_grad()
def embed(clip, sprites):
    """每张贴图的 CLIP 图像特征：灰底/白底、最近邻/双三次放大四种视图相加后归一化。"""
    device = clip.device
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)
    out = []
    for i in range(0, len(sprites), 256):
        total = 0
        for background in (128.0, 255.0):
            for resample in (Image.NEAREST, Image.BICUBIC):
                batch = []
                for rgba in sprites[i:i + 256]:
                    alpha = rgba[..., 3:].astype(np.float32) / 255.0
                    rgb = rgba[..., :3].astype(np.float32) * alpha + background * (1 - alpha)
                    batch.append(np.asarray(Image.fromarray(rgb.round().astype(np.uint8)).resize((224, 224), resample)))
                x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(device) / 255.0
                total = total + clip._norm(clip.model.get_image_features(pixel_values=(x - mean) / std))
        out.append(total / total.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def summary(label, ranks):
    r = np.asarray(ranks)
    print(f"{label:30} n={len(r):6}  top1 {np.mean(r <= 1):4.0%}  top5 {np.mean(r <= 5):4.0%}  top20 {np.mean(r <= 20):4.0%}  "
          f"median rank {int(np.median(r))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--train", type=Path, nargs="*", help="synthetic datasets; default: synth2, many, manual",
                    default=[ROOT / "dataset" / d for d in ("synth2", "many", "manual")])
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "_bind_cache.npz", help="features of the training sprites")
    ap.add_argument("--n", type=int, default=4, help="images per concept")
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--per-concept", type=Path, help="write each concept's ranks to this CSV")
    ap.add_argument("--names-rank", action="store_true", help="also report the name-based rank of eval_recall.py")
    args = ap.parse_args()
    dev = torch.device("cuda")

    concepts = load_concepts()
    zh_of = {c["en"]: c["zh"] for c in concepts}
    rows = []
    for folder in args.train:
        for line in (folder / "metadata.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["name_en"] in zh_of:
                rows.append((folder / r["file"], r["name_en"], r["sha1"]))
    names = sorted({name for _, name, _ in rows})
    index = {n: i for i, n in enumerate(names)}
    owner = torch.tensor([index[name] for _, name, _ in rows], device=dev)

    clip = Clip(dev, ["x"])
    key = hashlib.sha1("".join(sha for _, _, sha in rows).encode()).hexdigest()
    if args.cache.exists() and str(np.load(args.cache)["key"]) == key:
        feats = torch.from_numpy(np.load(args.cache)["feats"]).to(dev)
    else:
        feats = embed(clip, [np.asarray(Image.open(path).convert("RGBA")) for path, _, _ in rows])
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, key=key, feats=feats.cpu().numpy())
    sums = torch.zeros(len(names), feats.shape[1], device=dev).index_add_(0, owner, feats)
    count = torch.bincount(owner, minlength=len(names))
    centers = sums / sums.norm(dim=-1, keepdim=True)

    def ranks_of(f, own, leave_out=False):
        out = []
        for i in range(0, len(f), 4096):
            fi, oi = f[i:i + 4096], own[i:i + 4096]
            sims = fi @ centers.T
            if leave_out:
                # 训练图和"去掉它自己之后"的本物品平均值比，否则是自己和自己比
                rest = sums[oi] - fi
                mine = (fi * (rest / rest.norm(dim=-1, keepdim=True).clamp(min=1e-6))).sum(-1)
            else:
                mine = sims[torch.arange(len(fi)), oi]
            sims[torch.arange(len(fi)), oi] = -1.0
            out.append((sims > mine[:, None]).sum(1) + 1)
        return torch.cat(out).cpu().numpy()

    several = count[owner] > 1
    train_ranks = ranks_of(feats[several], owner[several], leave_out=True)
    summary("training sprites (ceiling)", train_ranks)

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    prompts = [f"{zh_of[n]}, {n}".lower() for n in names]
    ctx, mask = text.encode(prompts)
    null_ctx, null_mask = text.encode([""])
    tok = null_tok = None
    if cfg["model"].get("num_tokens"):
        token_map = TokenMap.load(run / "tokens.json")
        tok = token_map.encode(text.tok, prompts, cfg["max_len"], dev)
        null_tok = token_map.encode(text.tok, [""], cfg["max_len"], dev)
    del text.model
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(run / f"ema_{args.step:06d}.pt", map_location=dev, weights_only=True))
    ac = cosine_alphas_cumprod().to(dev)
    sprites = []
    g = torch.Generator(device=dev).manual_seed(0)
    for i in range(0, len(names), 64):
        rep = lambda x: x[i:i + 64].repeat_interleave(args.n, 0)
        with autocast(dev):
            x = ddim_sample(model, rep(ctx), rep(mask), null_ctx, null_mask, 16, 0, ac, steps=args.steps, guidance=args.guidance,
                            generator=g, source=1.0, tok=None if tok is None else rep(tok), null_tok=null_tok)
        sprites += list(to_rgba_uint8(x))
    del model
    torch.cuda.empty_cache()

    own = torch.arange(len(names), device=dev).repeat_interleave(args.n)
    model_ranks = ranks_of(embed(clip, sprites), own).reshape(len(names), args.n)
    label = f"{args.run} @ {args.step}"
    summary(label, model_ranks.ravel())
    med = np.median(model_ranks, axis=1)
    print(f"concepts whose typical image is the nearest concept: {np.mean(med <= 1):.0%}; within 5: {np.mean(med <= 5):.0%}; "
          f"beyond 50: {np.mean(med > 50):.0%}", flush=True)
    n_train = count.cpu().numpy()
    for lo, hi in ((1, 9), (10, 15), (16, 20), (21, 99)):
        pick = (n_train >= lo) & (n_train <= hi)
        if pick.any():
            summary(f"  {lo}-{hi} training sprites ({int(pick.sum())})", model_ranks[pick].ravel())

    name_ranks = None
    if args.names_rank:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        all_names = [c["en"] for c in concepts]
        where = {n: i for i, n in enumerate(all_names)}
        name_ranks = np.array(direct_ranks(clip, tokenizer, sprites, all_names, [where[n] for n in names for _ in range(args.n)]))
        print(f"name-based rank (eval_recall.py): median {int(np.median(name_ranks))}, top100 {np.mean(name_ranks <= 100):.0%}")
        name_ranks = name_ranks.reshape(len(names), args.n)

    if args.per_concept:
        by = {}
        for name_i, r in zip(owner[several].cpu().numpy(), train_ranks):
            by.setdefault(int(name_i), []).append(int(r))
        args.per_concept.parent.mkdir(parents=True, exist_ok=True)
        with args.per_concept.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name_en", "name_zh", "n_train", "train_median", "model_median", "model_best"]
                       + (["name_rank_median"] if name_ranks is not None else []))
            for i, n in enumerate(names):
                w.writerow([n, zh_of[n], int(n_train[i]), int(np.median(by[i])) if i in by else "", int(med[i]), int(model_ranks[i].min())]
                           + ([int(np.median(name_ranks[i]))] if name_ranks is not None else []))
        print(f"per-concept ranks: {args.per_concept}")


if __name__ == "__main__":
    main()
