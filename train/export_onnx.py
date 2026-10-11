"""把训练好的模型导出成客户端用的文件：unet.onnx、text_encoder.onnx、tokenizer.json、manifest.json。

The text encoder's vocabulary is pruned to Chinese + ASCII pieces and its weights are quantized to int8.
Requires onnx and onnxruntime in addition to the training dependencies.

Usage:
  python -X utf8 train/export_onnx.py --run bi16 --name SIC-1.0
  python -X utf8 train/export_onnx.py --run bi16 --ckpt runs/bi16/ema_020000.pt --no-quantize
  python -X utf8 train/export_onnx.py --run sic16 --names-run sic17 --name SIC-1.2   # two networks in one package
"""
import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import T, cosine_alphas_cumprod  # noqa: E402
from model import UNet  # noqa: E402
from train import PREVIEW_EN, PREVIEW_ZH, TokenMap, save_grid, to_rgba_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OPSET = 18
# 保留的字符范围：ASCII、常用标点、CJK 标点、汉字、全角字符
KEEP_RANGES = [(0x20, 0x7E), (0xB7, 0xB7), (0xD7, 0xD7), (0x2000, 0x206F), (0x3000, 0x303F), (0x4E00, 0x9FFF),
               (0xFF00, 0xFFEF)]


def keep_piece(piece):
    return all(any(lo <= ord(c) <= hi for lo, hi in KEEP_RANGES) for c in piece.replace("▁", ""))


def prune_tokenizer(name, max_len, out_path):
    """裁掉用不到的词表项，返回 (保留的旧 id 列表, 原始分词器, 裁剪后的分词器)。"""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    src = Path(hf_hub_download(name, "tokenizer.json"))
    spec = json.loads(src.read_text(encoding="utf-8"))
    assert spec["model"]["type"] == "Unigram", "vocabulary pruning is only implemented for Unigram tokenizers"
    vocab = spec["model"]["vocab"]
    special = {t["id"] for t in spec["added_tokens"]}
    keep = [i for i, (piece, _) in enumerate(vocab) if i in special or keep_piece(piece)]
    new_id = {old: new for new, old in enumerate(keep)}
    spec["model"]["vocab"] = [vocab[i] for i in keep]
    spec["model"]["unk_id"] = new_id[spec["model"]["unk_id"]]
    for t in spec["added_tokens"]:
        t["id"] = new_id[t["id"]]
    post = spec.get("post_processor") or {}
    for tok in (post.get("special_tokens") or {}).values():
        tok["ids"] = [new_id[i] for i in tok["ids"]]
    out_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    pruned = Tokenizer.from_file(str(out_path))
    pad = pruned.token_to_id("<pad>")
    # 把定长填充和截断写进文件，客户端加载后直接得到 max_len 个 id
    pruned.enable_padding(length=max_len, pad_id=pad, pad_token="<pad>")
    pruned.enable_truncation(max_length=max_len)
    pruned.save(str(out_path))
    original = Tokenizer.from_file(str(src))
    original.enable_padding(length=max_len, pad_id=original.token_to_id("<pad>"), pad_token="<pad>")
    original.enable_truncation(max_length=max_len)
    return keep, original, pruned


def check_tokenizer(original, pruned, keep, texts):
    new_id = {old: new for new, old in enumerate(keep)}
    unk = new_id[original.token_to_id("<unk>")]
    lost = mismatch = 0
    for a, b in zip(original.encode_batch(texts), pruned.encode_batch(texts)):
        mapped = [new_id.get(i, unk) for i in a.ids]
        lost += any(i not in new_id for i in a.ids)
        mismatch += mapped != b.ids
    return lost, mismatch


class TextWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def add_token_vectors(path, remap, table, output="last_hidden_state"):
    """把训练出来的"每个词一组向量"并进文本编码器：原来的特征 + table[remap[input_ids]]。

    这一步放在量化之后做，向量表保持 float32。output 是 last_hidden_state 时替换原来的输出（单网络的模型，
    客户端不用改）；是别的名字时原输出不动，另加一个输出（两个网络共用一个文本编码器的模型包）。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    plain = "last_hidden_state"
    if output == "last_hidden_state":
        plain = "e5_hidden_state"
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == "last_hidden_state":
                    node.output[i] = plain
    else:
        extra = onnx.ValueInfoProto()
        extra.CopyFrom(graph.output[0])
        extra.name = output
        graph.output.append(extra)
    graph.initializer.extend([numpy_helper.from_array(remap.astype(np.int64), "token_vector_index"),
                              numpy_helper.from_array(table.astype(np.float32), "token_vector_table")])
    graph.node.extend([
        helper.make_node("Gather", ["token_vector_index", "input_ids"], ["token_vector_rows"], name="token_vector_rows"),
        helper.make_node("Gather", ["token_vector_table", "token_vector_rows"], ["token_vectors"], name="token_vectors"),
        helper.make_node("Add", [plain, "token_vectors"], [output], name="add_token_vectors"),
    ])
    assert graph.output[0].name == "last_hidden_state" and graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_aliases(folders, mixes, hand_drawn_names, has_source=True, alias_source=1.0):
    """客户端的名称补全表：物品的各种叫法 -> ["中文名, 英文名", 画风混合比例]。

    补全写法和训练时一致；混合比例来自 pick_sources.py，表示这个物品用多少"合成图"的画风画得更像。
    """
    table = {}
    rows = [json.loads(line) for folder in folders for line in (folder / "metadata.jsonl").open(encoding="utf-8")]
    sprites = Counter(f"{r['name_zh']}, {r['name_en']}".lower() for r in rows)
    pair_of = {r["name_en"].lower(): f"{r['name_zh']}, {r['name_en']}".lower() for r in rows}
    # 手工补的通用叫法（"相机" -> film camera）优先级最高
    extra = ROOT / "synth" / "concepts" / "extra_aliases.tsv"
    if extra.exists() and not mixes:
        for line in extra.read_text(encoding="utf-8").splitlines():
            key, _, name = line.partition("\t")
            if name.strip().lower() in pair_of:
                table[key.strip().lower()] = [pair_of[name.strip().lower()], alias_source if has_source else 0.0]
    # 模组里天天出现的装备词不补全，见 no_alias.txt 开头的说明
    blocked = ROOT / "synth" / "concepts" / "no_alias.txt"
    blocked = {line.strip().lower() for line in blocked.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")} if blocked.exists() else set()
    # 同一个叫法被多个物品共用时，归训练图最多的那个
    for r in sorted(rows, key=lambda r: -sprites[f"{r['name_zh']}, {r['name_en']}".lower()]):
        pair = f"{r['name_zh']}, {r['name_en']}".lower()
        for key in [r["name_zh"], r["name_en"], *r.get("alt_names_zh", []), *r["alt_names_en"]]:
            key = key.lower().strip()
            # 太短的叫法容易误伤别的词
            if key not in blocked and len(key) >= (2 if any("\u4e00" <= ch <= "\u9fff" for ch in key) else 3):
                # 模组贴图里有同名物品时一般不补全，模型直接认识这个名字。例外：同名的手绘贴图只有两三张
                # （模型记不住），而合成数据里这个物品有十张以上，这时用合成的更可靠（如"雨伞"）
                if not mixes and (hand_drawn_names.get(key, 0) > 3 or (key in hand_drawn_names and sprites[pair] < 10)):
                    continue
                # 没有逐个物品挑过画风时，合成数据里的物品一律按合成图的画风出
                mix, diff = mixes.get(pair, [alias_source if has_source else 0.0, 0.0])
                if key in hand_drawn_names and mixes:
                    # 模组贴图里本来就有同名物品：除非合成风格明显更像，否则保持手绘风格
                    mix = max(m for m, _ in mixes.values()) if diff > 0.01 else 0.0
                table.setdefault(key, [pair, mix])
    return table


def build_coverage(folders):
    """每个合成物品（"中文名, 英文名"）的训练图数量。给调用模型的程序判断"这个名字模型到底熟不熟"用。"""
    rows = [json.loads(line) for folder in folders for line in (folder / "metadata.jsonl").open(encoding="utf-8")]
    return dict(Counter(f"{r['name_zh']}, {r['name_en']}".lower() for r in rows).most_common())


def ort_sample(unet, ctx, mask, null_ctx, null_mask, size, size_idx, steps, guidance, seed, source=None):
    """纯 numpy 的 DDIM 采样，和客户端的实现逐步对应。"""
    ac = cosine_alphas_cumprod().double().numpy()
    b = ctx.shape[0]
    x = np.random.default_rng(seed).standard_normal((b, 4, size, size)).astype(np.float32)
    ts = np.round(np.linspace(T - 1, 0, steps)).astype(np.int64)
    ctx2 = np.concatenate([ctx, np.repeat(null_ctx, b, 0)])
    mask2 = np.concatenate([mask, np.repeat(null_mask, b, 0)])
    feed = {"ctx": ctx2, "mask": mask2, "size_idx": np.full(2 * b, size_idx, dtype=np.int64)}
    if source is not None:
        feed["src_mix"] = np.full(2 * b, source, dtype=np.float32)
    for i, t in enumerate(ts):
        v = unet.run(None, {"x": np.concatenate([x, x]), "t": np.full(2 * b, t, dtype=np.int64), **feed})[0]
        v = v[b:] + guidance * (v[:b] - v[b:])
        a, s = np.sqrt(ac[t]), np.sqrt(1 - ac[t])
        x0 = np.clip(a * x - s * v, -1, 1)
        if i == steps - 1:
            return x0.astype(np.float32)
        eps = (x - a * x0) / s
        ap = ac[ts[i + 1]]
        x = (np.sqrt(ap) * x0 + np.sqrt(1 - ap) * eps).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Export the model for the client.")
    ap.add_argument("--run", default="bi16")
    ap.add_argument("--ckpt", help="EMA weights; default is the latest ema_*.pt in the run")
    ap.add_argument("--name", help="model name written to the manifest, e.g. SIC-1.0; default <run>-<step>")
    ap.add_argument("--out", type=Path, help="output directory; default export/<name>")
    ap.add_argument("--names-run", help="a second run to ship next to --run as the name-bound network (it must use token "
                    "vectors); the package then holds unet.onnx, unet_names.onnx and one shared text encoder")
    ap.add_argument("--names-ckpt", help="EMA weights of --names-run; default is its latest ema_*.pt")
    ap.add_argument("--no-quantize", action="store_true", help="keep the text encoder in float32")
    ap.add_argument("--steps", type=int, default=30, help="default sampling steps written to the manifest")
    ap.add_argument("--guidance", type=float, default=5.5, help="default CFG scale written to the manifest")
    ap.add_argument("--alias-source", type=float, default=1.0,
                    help="style mix for names that come from the synthetic dataset (when pick_sources.py was not run)")
    ap.add_argument("--source", type=float, default=0.0, help="default style mix written to the manifest (0 = hand-drawn)")
    args = ap.parse_args()

    import onnxruntime as ort
    from transformers import AutoModel

    run = ROOT / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ckpt = Path(args.ckpt) if args.ckpt else sorted(run.glob("ema_*.pt"))[-1]
    name = args.name or f"{args.run}-{ckpt.stem.split('_')[-1]}"
    out = args.out or ROOT / "export" / name
    out.mkdir(parents=True, exist_ok=True)
    max_len, dim = cfg["max_len"], cfg["model"]["ctx_dim"]

    # 1. UNet
    unet = UNet(**cfg["model"]).eval()
    unet.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    ex_mask = torch.zeros(2, max_len, dtype=torch.bool)
    ex_mask[:, :4] = True
    has_source = bool(cfg["model"].get("num_sources"))
    example = (torch.randn(2, 4, 16, 16), torch.tensor([10, 500]), torch.randn(2, max_len, dim), ex_mask,
               torch.zeros(2, dtype=torch.long))
    names = ["x", "t", "ctx", "mask", "size_idx"]
    if has_source:
        example += (torch.tensor([0.0, 0.7]),)
        names.append("src_mix")
    axes = {n: {0: "batch"} for n in names}
    axes["x"] = axes["v"] = {0: "batch", 2: "height", 3: "width"}
    torch.onnx.export(unet, example, str(out / "unet.onnx"), dynamo=False, opset_version=OPSET,
                      input_names=names, output_names=["v"], dynamic_axes=axes)
    print(f"unet.onnx {(out / 'unet.onnx').stat().st_size / 1e6:.1f} MB (from {ckpt.name})", flush=True)

    # 第二个网络：认名字更准的那个，和第一个共用分词器、文本编码器和补全表
    names_unet = names_run = names_ckpt = None
    if args.names_run:
        names_run = ROOT / "runs" / args.names_run
        cfg2 = json.loads((names_run / "config.json").read_text(encoding="utf-8"))
        assert not cfg["model"].get("num_tokens") and cfg2["model"].get("num_tokens"), \
            "--run must be a model without token vectors and --names-run one with them"
        same = ("text_encoder", "max_len", "sizes", "prediction", "schedule", "timesteps")
        assert all(cfg[k] == cfg2[k] for k in same) and bool(cfg2["model"].get("num_sources")) == has_source, \
            "the two runs must share the text encoder, prompt length, sizes and style input"
        names_ckpt = Path(args.names_ckpt) if args.names_ckpt else sorted(names_run.glob("ema_*.pt"))[-1]
        names_unet = UNet(**cfg2["model"]).eval()
        names_unet.load_state_dict(torch.load(names_ckpt, map_location="cpu", weights_only=True))
        torch.onnx.export(names_unet, example, str(out / "unet_names.onnx"), dynamo=False, opset_version=OPSET,
                          input_names=names, output_names=["v"], dynamic_axes=axes)
        print(f"unet_names.onnx {(out / 'unet_names.onnx').stat().st_size / 1e6:.1f} MB (from {args.names_run}/{names_ckpt.name})", flush=True)

    # 2. 分词器和文本编码器：裁词表后导出
    keep, tok_orig, tok = prune_tokenizer(cfg["text_encoder"], max_len, out / "tokenizer.json")
    rows = [json.loads(line) for line in (ROOT / "dataset" / "clean" / "metadata.jsonl").open(encoding="utf-8")]
    texts = sorted({n.lower() for r in rows for n in [r["name_en"], r["name_zh"], *r["alt_names_en"]] if n})
    lost, mismatch = check_tokenizer(tok_orig, tok, keep, texts)
    print(f"tokenizer: kept {len(keep)} pieces; on {len(texts)} dataset names: "
          f"{lost} contain a dropped piece, {mismatch} tokenize differently", flush=True)

    enc = AutoModel.from_pretrained(cfg["text_encoder"]).eval()
    full = enc.get_input_embeddings().weight.detach()
    enc.set_input_embeddings(torch.nn.Embedding.from_pretrained(full[torch.tensor(keep)].clone(), freeze=True))
    enc.config.vocab_size = len(keep)
    wrap = TextWrap(enc)
    probe = PREVIEW_EN + PREVIEW_ZH + [""] + texts[::40]
    batch = tok.encode_batch(probe)
    ids = torch.tensor([e.ids for e in batch])
    att = torch.tensor([e.attention_mask for e in batch])
    with torch.no_grad():
        ref = wrap(ids, att).numpy()
    # 带词向量的模型：把分词器的编号换成训练时的紧凑编号（特殊符号和没见过的词是 0 号，向量为零）
    remap = table = ref_names = None
    vector_owner = (run, unet) if cfg["model"].get("num_tokens") else (names_run, names_unet) if names_unet is not None else None
    if vector_owner:
        index = TokenMap.load(vector_owner[0] / "tokens.json").index
        remap = np.array([index.get(old, 0) for old in keep])
        table = vector_owner[1].tok_emb.weight.detach().numpy()
        if names_unet is None:
            ref = ref + table[remap[ids.numpy()]]
        else:
            ref_names = ref + table[remap[ids.numpy()]]
        print(f"token vectors: {len(index)} trained, {int((remap > 0).sum())} of them in the exported vocabulary", flush=True)
    fp32 = out / "text_encoder_fp32.onnx"
    torch.onnx.export(wrap, (ids[:2], att[:2]), str(fp32), dynamo=False, opset_version=OPSET,
                      input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "length"}, "attention_mask": {0: "batch", 1: "length"},
                                    "last_hidden_state": {0: "batch", 1: "length"}})
    final = out / "text_encoder.onnx"
    if args.no_quantize:
        fp32.replace(final)
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        # 全部算子做 int8 动态量化：特征和原模型的余弦约 0.99，并排出图看不出差别，体积是只量化词向量表的一半
        quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QUInt8)
        fp32.unlink()
    if table is not None:
        add_token_vectors(final, remap, table, "last_hidden_state" if names_unet is None else "names_hidden_state")
    print(f"text_encoder.onnx {final.stat().st_size / 1e6:.1f} MB", flush=True)

    # 3. 用 onnxruntime 验证：文本特征和 PyTorch 对比，再完整采样一遍
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    text_sess = ort.InferenceSession(str(final), opts, providers=["CPUExecutionProvider"])
    unet_sess = ort.InferenceSession(str(out / "unet.onnx"), opts, providers=["CPUExecutionProvider"])
    text_out = text_sess.run(None, {"input_ids": ids.numpy(), "attention_mask": att.numpy()})
    got = text_out[0]
    m = att.numpy().astype(bool)
    cos = (got[m] * ref[m]).sum(-1) / (np.linalg.norm(got[m], axis=-1) * np.linalg.norm(ref[m], axis=-1))
    print(f"text features vs PyTorch: cosine mean {cos.mean():.4f}, min {cos.min():.4f}", flush=True)
    if names_unet is not None:
        got_names = text_out[1]
        cos = (got_names[m] * ref_names[m]).sum(-1) / (np.linalg.norm(got_names[m], axis=-1) * np.linalg.norm(ref_names[m], axis=-1))
        print(f"text features with token vectors vs PyTorch: cosine mean {cos.mean():.4f}, min {cos.min():.4f}", flush=True)
    with torch.no_grad():
        v_ref = unet(*example).numpy()
    v_got = unet_sess.run(None, {n: e.numpy() for n, e in zip(names, example)})[0]
    print(f"unet vs PyTorch: max abs diff {np.abs(v_got - v_ref).max():.2e}", flush=True)

    n_prev = len(PREVIEW_EN) + len(PREVIEW_ZH)
    null = n_prev  # probe 里第 n_prev 个是空提示词
    mask_np = att.numpy().astype(bool)
    reps = 3
    for label, feats in (("onnx", got), ("torch_text", ref)):
        idx = np.repeat(np.arange(n_prev), reps)
        x = ort_sample(unet_sess, feats[idx], mask_np[idx], feats[null:null + 1], mask_np[null:null + 1], 16, 0,
                       args.steps, args.guidance, seed=0, source=args.source if has_source else None)
        save_grid(to_rgba_uint8(torch.from_numpy(x)), reps * 4, out / f"check_{label}.png")
    print(f"sample grids: {out / 'check_onnx.png'} (exported text encoder) vs check_torch_text.png (float32 text features)")
    if names_unet is not None:
        names_sess = ort.InferenceSession(str(out / "unet_names.onnx"), opts, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            v_ref = names_unet(*example).numpy()
        v_got = names_sess.run(None, {n: e.numpy() for n, e in zip(names, example)})[0]
        print(f"unet_names vs PyTorch: max abs diff {np.abs(v_got - v_ref).max():.2e}", flush=True)
        idx = np.repeat(np.arange(n_prev), reps)
        x = ort_sample(names_sess, got_names[idx], mask_np[idx], got_names[null:null + 1], mask_np[null:null + 1], 16, 0,
                       args.steps, args.guidance, seed=0, source=args.source)
        save_grid(to_rgba_uint8(torch.from_numpy(x)), reps * 4, out / "check_onnx_names.png")

    # 4. manifest
    # 名称补全表：只有训练时用了合成数据才有
    extra = [Path(p) for p in cfg["args"].get("extra") or []]
    mixes_path = run / "concept_sources.json"
    mixes = json.loads(mixes_path.read_text(encoding="utf-8")) if mixes_path.exists() else {}
    hand_counts = Counter(n.lower() for r in rows for n in [r["name_en"], r["name_zh"], *r["alt_names_en"]] if n)
    aliases = build_aliases(extra, mixes, hand_counts, has_source, args.alias_source)
    (out / "aliases.json").write_text(json.dumps(aliases, ensure_ascii=False), encoding="utf-8")
    # 可选文件，不进 manifest：客户端不用它，命令行脚本和 AI 技能会读
    (out / "coverage.json").write_text(json.dumps(build_coverage(extra), ensure_ascii=False), encoding="utf-8")
    print(f"aliases.json: {len(aliases)} names, {sum(1 for v in aliases.values() if v[1] > 0)} of them use the synthetic style", flush=True)

    listed = [("unet", "unet.onnx"), ("text_encoder", "text_encoder.onnx"), ("tokenizer", "tokenizer.json"), ("aliases", "aliases.json")]
    if names_unet is not None:
        # 手绘训练图的名字：提示词和其中一个完全相同时，交给认名字的那个网络
        known = sorted({n.lower() for r in rows if r["size"] == 16
                        for n in [r["name_en"], r["name_zh"], *r["alt_names_en"], *r.get("alt_names_zh", [])] if n})
        (out / "known_names.json").write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
        print(f"known_names.json: {len(known)} names", flush=True)
        listed += [("unet_names", "unet_names.onnx"), ("known_names", "known_names.json")]
    files = {key: {"name": name, "size": (out / name).stat().st_size, "sha256": sha256(out / name)} for key, name in listed}
    manifest = {
        "format": 1,
        "model_version": name,
        "source": f"{args.run}/{ckpt.name}" + (f" + {args.names_run}/{names_ckpt.name}" if names_unet is not None else ""),
        "files": files,
        "text": {"encoder": cfg["text_encoder"], "max_len": max_len, "dim": dim, "lowercase": True,
                 **({"names_output": "names_hidden_state"} if names_unet is not None else {})},
        "diffusion": {"timesteps": T, "schedule": "cosine", "prediction": "v"},
        "sizes": {"16": 0, "32": 1},
        "trained_sizes": cfg["sizes"],
        "has_source": has_source,
        "defaults": {"steps": args.steps, "guidance": args.guidance, "size": 16, "count": 3, "source": args.source},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f["size"] for f in files.values())
    print(f"manifest.json written; total download size {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
