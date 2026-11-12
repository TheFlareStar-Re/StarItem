"""按描述画得准不准：拿训练时留出的描述去出图，看出来的图是不是更像那句描述原本对应的那张贴图。

Training with --captions keeps the descriptions of a few percent of the sprites out (train.held_out). For such a sprite
the model draws from its description, and the result is compared by CLIP image similarity with all training sprites of
the same item. A model that only knows names cannot tell the described sprite from its siblings, so its position among
them is random (0.5 on the 0..1 scale used here); a model that follows the colors and parts in the description puts it
near the front (towards 0). The same is done with the plain name as the prompt, which is the reference.

Only the generated sprites of the datasets "many" and "synth2" are used: the redraws in "manual" come in groups of
near copies whose descriptions are nearly the same, so a held-out one is not really unseen.

Usage:
  .venv/Scripts/python.exe -X utf8 train/eval_desc.py --run arm_both --step 60000
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "synth"))
from diffusion import cosine_alphas_cumprod, ddim_sample  # noqa: E402
from eval_bind import embed  # noqa: E402
from generate import load_concepts  # noqa: E402
from model import UNet  # noqa: E402
from pilot import Clip  # noqa: E402
from train import TextEncoder, TokenMap, autocast, description_prompts, held_out, load_captions, to_rgba_uint8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--captions", type=Path, default=ROOT / "dataset" / "captions.jsonl")
    ap.add_argument("--holdout", type=int, default=3, help="must match --desc-holdout of the training run")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "_bind_cache.npz", help="CLIP features written by eval_bind.py")
    ap.add_argument("--count", type=int, default=1000, help="held-out sprites to test")
    ap.add_argument("--n", type=int, default=2, help="images per prompt")
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")

    concepts = load_concepts()
    zh_of = {c["en"]: c["zh"] for c in concepts}
    # 和 eval_bind.py 同样的顺序，缓存里的特征才能对上
    rows = []
    for name in ("synth2", "many", "manual"):
        for line in (ROOT / "dataset" / name / "metadata.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["name_en"] in zh_of:
                rows.append((name, r))
    cache = np.load(args.cache)
    key = hashlib.sha1("".join(r["sha1"] for _, r in rows).encode()).hexdigest()
    assert str(cache["key"]) == key, "run train/eval_bind.py once first: its feature cache does not match the datasets"
    feats = torch.from_numpy(cache["feats"]).to(dev)

    members = {}
    for i, (_, r) in enumerate(rows):
        members.setdefault(r["name_en"], []).append(i)
    captions = load_captions(args.captions)
    cases = []
    for i, (name, r) in enumerate(rows):
        caption = captions.get((name, r["file"]))
        if name != "manual" and caption and held_out(r["sha1"], args.holdout) and len(members[r["name_en"]]) >= 10:
            cases.append((i, description_prompts(r, caption, "zh")[0], f"{r['name_zh']}, {r['name_en']}".lower()))
    random.Random(0).shuffle(cases)
    cases = cases[:args.count]
    print(f"{len(cases)} held-out sprites, e.g. {cases[0][1]!r} (name prompt {cases[0][2]!r})", flush=True)

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    text = TextEncoder(cfg["text_encoder"], cfg["max_len"], dev)
    token_map = TokenMap.load(run / "tokens.json") if cfg["model"].get("num_tokens") else None
    model = UNet(**cfg["model"]).to(dev).eval()
    model.load_state_dict(torch.load(run / f"ema_{args.step:06d}.pt", map_location=dev, weights_only=True))
    ac = cosine_alphas_cumprod().to(dev)
    clip = Clip(dev, ["x"])
    null_ctx, null_mask = text.encode([""])
    null_tok = token_map.encode(text.tok, [""], cfg["max_len"], dev) if token_map else None

    def position(prompts):
        """每张生成图：留出的那张贴图在同物品的贴图里按相似度排在多靠前（0 = 最像，1 = 最不像），以及两者的相似度。"""
        ctx, mask = text.encode(prompts)
        tok = token_map.encode(text.tok, prompts, cfg["max_len"], dev) if token_map else None
        g = torch.Generator(device=dev).manual_seed(0)
        sprites = []
        for i in range(0, len(prompts), 64):
            rep = lambda x: x[i:i + 64].repeat_interleave(args.n, 0)
            with autocast(dev):
                x = ddim_sample(model, rep(ctx), rep(mask), null_ctx, null_mask, 16, 0, ac, steps=args.steps, guidance=args.guidance,
                                generator=g, source=1.0, tok=None if tok is None else rep(tok), null_tok=null_tok)
            sprites += list(to_rgba_uint8(x))
        made = embed(clip, sprites)
        where, close = [], []
        for j, (target, _, _) in enumerate(cases):
            group = torch.tensor(members[rows[target][1]["name_en"]], device=dev)
            sims = made[j * args.n:(j + 1) * args.n] @ feats[group].T
            own = sims[:, (group == target).nonzero()[0, 0]]
            where += (((sims > own[:, None]).sum(1)) / (len(group) - 1)).tolist()
            close += own.tolist()
        return np.array(where), np.array(close)

    label = f"{args.run} @ {args.step}"
    for kind, prompts in (("description", [c[1] for c in cases]), ("name only", [c[2] for c in cases])):
        where, close = position(prompts)
        print(f"{label} | {kind:11} | position among siblings: mean {where.mean():.3f}, median {np.median(where):.3f} (0.5 = chance) | "
              f"it is the closest sibling: {np.mean(where == 0):.0%} | similarity to it {close.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
