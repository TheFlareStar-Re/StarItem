#!/usr/bin/env python3
"""Generate Minecraft-style 16x16 item sprites with the StarItem SIC model (ONNX, CPU).

Usage:
  python generate.py "alarm clock" --out sprites
  python generate.py "红宝石镐" "golden crown" --out sprites --count 3 --seed 7
  python generate.py "red mug" --ref photo_of_my_mug.png --out sprites
  python generate.py --search camera
  python generate.py --preview-of sprites/a_1_2.png sprites/b_7_1.png --out sprites

For each prompt this writes <out>/<name>_<seed>_<n>.png (16x16 RGBA, transparent background) and one
<out>/<name>_<seed>_preview.png that shows the sprites enlarged side by side, then prints one JSON line.

Needs: onnxruntime, numpy, tokenizers (and Pillow, only for --ref). The model folder (manifest.json, unet.onnx, text_encoder.onnx,
tokenizer.json, aliases.json, optionally unet_names.onnx, known_names.json, coverage.json and matting.onnx) is found through --models,
the STARITEM_MODELS environment variable, ./models, ~/.staritem/models, or a models folder next to this skill.

From SIC-1.2 on the folder holds two networks, nicknamed Mo and Jang. Jang (unet_names.onnx) draws items it knows by
name much more reliably; Mo (unet.onnx) follows colors and unusual combinations more readily. The script picks per prompt.
For anything that is not exactly a known name it mostly uses a relay: Mo draws the first fifth of the sampling steps, where
the outline is decided, and Jang finishes, which keeps Mo's steadier shapes and Jang's colors.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import zlib
from pathlib import Path

REQUIRED = ("manifest.json", "unet.onnx", "text_encoder.onnx", "tokenizer.json")
RELAY = 0.2  # share of the sampling steps Mo draws before Jang takes over
MIXED = ("relay", "mo", "relay", "jang")  # who draws candidate k when the prompt is not exactly a known name


def fail(message):
    print(json.dumps({"error": message}, ensure_ascii=False))
    sys.exit(1)


try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError as e:
    fail(f"missing Python package '{e.name}'. Install with: pip install onnxruntime numpy tokenizers")


def find_models(explicit):
    here = Path(__file__).resolve().parent.parent
    candidates = [explicit, os.environ.get("STARITEM_MODELS"), "models", Path.home() / ".staritem" / "models", here / "models"]
    tried = []
    for c in candidates:
        if not c:
            continue
        folder = Path(c).expanduser()
        tried.append(str(folder))
        if all((folder / name).is_file() for name in REQUIRED):
            return folder
    fail("model folder not found. It must contain " + ", ".join(REQUIRED) + ". Looked in: " + "; ".join(tried)
         + ". Pass --models <folder> or set STARITEM_MODELS.")


def is_cjk(text):
    return any("一" <= ch <= "鿿" for ch in text)


def ends_with_word(text, key):
    """True when the prompt ends with this name. ASCII names must start at a word boundary."""
    if not text.endswith(key):
        return False
    head = text[:len(text) - len(key)]
    return is_cjk(key) or not head or not head[-1].isalnum()


def alias_key(text, aliases):
    """The longest known everyday-object name the prompt ends with, or None."""
    found = [k for k in aliases if ends_with_word(text, k)]
    return max(found, key=len) if found else None


def expand(text, aliases):
    """Name completion, same rule as the desktop client: returns (prompt for the model, style or None)."""
    key = alias_key(text, aliases)
    if key is None:
        return text, None
    pair, style = aliases[key]
    if key == text and text in pair.split(", "):
        return pair, style
    return (text if pair in text else f"{text}, {pair}"), style


def write_png(path, rgba):
    """Minimal PNG writer for an (H, W, 4) uint8 array, so Pillow is not needed."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def read_png(path):
    """Reads an 8-bit RGBA, non-interlaced PNG (the kind this script writes) into an (H, W, 4) uint8 array."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"{path} is not a PNG file")
    pos, idat, header = 8, b"", None
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    w, h, depth, color, _, _, interlace = header
    if (depth, color, interlace) != (8, 6, 0):
        fail(f"{path}: only 8-bit RGBA PNG files made by this script can be previewed")
    raw = zlib.decompress(idat)
    out = np.zeros((h, w * 4), dtype=np.uint8)
    stride = w * 4
    for y in range(h):
        kind = raw[y * (stride + 1)]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=y * (stride + 1) + 1).astype(np.int32)
        up = out[y - 1].astype(np.int32) if y else np.zeros(stride, dtype=np.int32)
        if kind in (1, 3, 4):  # filters that depend on the pixel to the left need a running pass
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                left = cur[i - 4] if i >= 4 else 0
                if kind == 1:
                    pred = left
                elif kind == 3:
                    pred = (left + up[i]) // 2
                else:
                    corner = up[i - 4] if i >= 4 else 0
                    base = left + up[i] - corner
                    pa, pb, pc = abs(base - left), abs(base - up[i]), abs(base - corner)
                    pred = left if pa <= pb and pa <= pc else (up[i] if pb <= pc else corner)
                cur[i] = (line[i] + pred) & 255
            line = cur
        elif kind == 2:
            line = (line + up) & 255
        out[y] = line
    return out.reshape(h, w, 4)


def preview(sprites, scale):
    """Sprites side by side, enlarged without smoothing, on a light checkerboard so transparency is visible."""
    n, size = len(sprites), sprites[0].shape[0]
    gap = 2
    sheet = np.full((size + 2 * gap, n * (size + gap) + gap, 4), 255, dtype=np.uint8)
    sheet[..., :3] = 60
    yy, xx = np.mgrid[0:size, 0:size]
    board = np.where(((yy // 2 + xx // 2) % 2 == 0)[..., None], 236, 214).astype(np.uint8)
    for i, sprite in enumerate(sprites):
        opaque = sprite[..., 3:] > 0
        x0 = gap + i * (size + gap)
        sheet[gap:gap + size, x0:x0 + size, :3] = np.where(opaque, sprite[..., :3], board)
    return sheet.repeat(scale, axis=0).repeat(scale, axis=1)


def load_coverage(folder):
    """Number of training sprites per known everyday object; empty when the model folder has no coverage.json."""
    path = folder / "coverage.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def full_name_in(expanded, coverage):
    """The known full name at the end of an expanded prompt, if any."""
    found = [name for name in coverage if expanded == name or expanded.endswith(", " + name)]
    return max(found, key=len) if found else None


def matting_mask(rgb, model_path):
    """Foreground mask of an (H, W, 3) image from the U^2-Net-p salient-object model."""
    from PIL import Image

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    h, w = rgb.shape[:2]
    x = np.asarray(Image.fromarray(rgb.astype(np.uint8), "RGB").resize((320, 320), Image.BILINEAR)).astype(np.float32)
    x = x / max(float(x.max()), 1e-6)
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pred = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    # Nearest sampling back to the picture's size; the mask only has to be right at 16x16 in the end.
    ys = np.minimum((np.arange(h) * 320) // h, 319)
    xs = np.minimum((np.arange(w) * 320) // w, 319)
    return pred[ys][:, xs] >= 0.5


def main_parts(mask, ratio=0.05):
    """Drops blobs much smaller than the largest one: bits of other things the matting model also marked."""
    h, w = mask.shape
    cells = mask.ravel().tolist()
    label = [0] * (h * w)
    sizes = [0]
    for start in range(h * w):
        if not cells[start] or label[start]:
            continue
        n = len(sizes)
        label[start] = n
        stack, count = [start], 0
        while stack:
            i = stack.pop()
            count += 1
            x = i % w
            for j in (i - 1 if x > 0 else -1, i + 1 if x + 1 < w else -1, i - w, i + w):
                if 0 <= j < h * w and cells[j] and not label[j]:
                    label[j] = n
                    stack.append(j)
        sizes.append(count)
    keep = np.array(sizes) >= max(ratio * max(sizes), 1)
    return keep[np.array(label).reshape(h, w)]


def load_reference(path, outline=True, tolerance=40.0, matting=None, info=None):
    """Turns a picture of one object into a 16x16 RGBA draft for the model.

    A transparent or plain background is removed directly. For any other background the object is cut out with
    the matting model, when `matting` points to its file. `info`, if given, receives how the background was removed.
    """
    info = {} if info is None else info
    info["background"] = "transparent"
    try:
        from PIL import Image
    except ImportError:
        fail("--ref needs the Pillow package. Install with: pip install pillow")
    try:
        im = Image.open(path).convert("RGBA")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read the reference image {path}: {e}")
    # Detail finer than this cannot survive in 16x16, and a small image keeps the background search fast.
    if max(im.size) > 256:
        k = math.ceil(max(im.size) / 256)
        im = im.reduce(k) if (np.asarray(im)[..., 3] == 255).all() else im.resize((im.width // k, im.height // k), Image.NEAREST)
    a = np.asarray(im).astype(np.float32)
    rgb, alpha = a[..., :3], a[..., 3]
    if (alpha < 128).mean() < 0.02:
        # No transparency: treat the border color as background and remove everything of that color connected to the border.
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0)
        if (np.linalg.norm(border - bg, axis=1) <= tolerance).mean() >= 0.85:
            info["background"] = "plain"
            close = np.linalg.norm(rgb - bg, axis=2) <= tolerance
            reach = np.zeros_like(close)
            reach[0], reach[-1], reach[:, 0], reach[:, -1] = close[0], close[-1], close[:, 0], close[:, -1]
            while True:
                grown = reach.copy()
                grown[1:] |= reach[:-1]
                grown[:-1] |= reach[1:]
                grown[:, 1:] |= reach[:, :-1]
                grown[:, :-1] |= reach[:, 1:]
                grown &= close
                if (grown == reach).all():
                    break
                reach = grown
            alpha = np.where(reach, 0.0, 255.0)
        elif matting is not None and Path(matting).is_file():
            # A photo: let the salient-object model decide what the object is.
            info["background"] = "model"
            alpha = np.where(main_parts(matting_mask(rgb, matting)), 255.0, 0.0)
        else:
            fail("the reference image has a busy background, and the model folder has no matting.onnx to cut the object out "
                 "with. Ask the user for a cut-out or a picture on a plain background, or add matting.onnx to the model folder.")
    opaque = alpha >= 128
    if not 0.01 <= opaque.mean() <= 0.97:
        fail("could not separate an object from the background of the reference image "
             f"({opaque.mean():.0%} of it would count as the object)")
    ys, xs = np.where(opaque)
    rgb, alpha = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = alpha.shape
    side = max(h, w)
    square = np.zeros((side, side, 4), dtype=np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    hard = np.where(alpha >= 128, 255.0, 0.0)
    square[y0:y0 + h, x0:x0 + w, :3] = rgb * (hard[..., None] / 255.0)  # premultiplied, so the background does not bleed in
    square[y0:y0 + h, x0:x0 + w, 3] = hard
    small = np.asarray(Image.fromarray(square.round().astype(np.uint8), "RGBA").resize((16, 16), Image.BOX)).astype(np.float32)
    cover = small[..., 3] / 255.0
    opaque = cover >= 0.5
    # Pillow already averages the color over the opaque part only when it resizes an RGBA image.
    color = np.where(opaque[..., None], small[..., :3], 0.0)
    if outline:
        # Item sprites have a darker rim; darkening the draft's edge cells steers the model towards that look.
        pad = np.pad(opaque, 1)
        inside = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        color = np.where((opaque & ~inside)[..., None], color * 0.55, color)
    out = np.zeros((16, 16, 4), dtype=np.uint8)
    out[..., :3] = np.clip(color, 0, 255).round()
    out[..., 3] = np.where(opaque, 255, 0)
    return out


def keep_colors(sprite, draft, amount):
    """Pulls a generated sprite's hues towards the reference draft while keeping the sprite's own light and dark shading."""
    both = (sprite[..., 3] > 0) & (draft[..., 3] > 0)
    if amount <= 0 or not both.any():
        return sprite
    to_ycc = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    a, b = sprite[..., :3].astype(np.float64) @ to_ycc.T, draft[..., :3].astype(np.float64) @ to_ycc.T
    # The draft's rim was darkened, which also shrank its chroma; scale it back by the brightness ratio before borrowing it.
    gain = np.clip(a[..., :1] / np.maximum(b[..., :1], 1.0), 0.0, 2.0)
    a[..., 1:] = np.where(both[..., None], (1 - amount) * a[..., 1:] + amount * b[..., 1:] * gain, a[..., 1:])
    out = sprite.copy()
    out[..., :3] = np.clip(a @ np.linalg.inv(to_ycc).T, 0, 255).round()
    out[sprite[..., 3] == 0] = 0
    return out


class Model:
    def __init__(self, folder):
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        path = folder / "aliases.json"
        self.aliases = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self.coverage = load_coverage(folder)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.opts = opts
        # Networks are loaded when first used; a prompt often needs only one of them.
        self.paths = {"mo": folder / "unet.onnx"}
        self.unets = {}
        self.known = set()
        self.names_output = self.manifest["text"].get("names_output")
        if self.names_output and (folder / "unet_names.onnx").is_file():
            self.paths["jang"] = folder / "unet_names.onnx"
            path = folder / "known_names.json"
            self.known = set(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else set()
        self.text = ort.InferenceSession(str(folder / "text_encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        # The exported tokenizer already truncates and pads to the model's length; use its attention mask as is.
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        steps = self.manifest["diffusion"]["timesteps"]
        x = np.linspace(0, steps, steps + 1, dtype=np.float64)
        f = np.cos((x / steps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = np.minimum(1 - f[1:] / f[:-1], 0.999)
        self.alphas = np.cumprod(1 - betas).astype(np.float32).astype(np.float64)
        self.has_style = bool(self.manifest.get("has_source"))

    def unet(self, network):
        if network not in self.unets:
            self.unets[network] = ort.InferenceSession(str(self.paths[network]), self.opts, providers=["CPUExecutionProvider"])
        return self.unets[network]

    def encode(self, prompt):
        """Text features per network (Jang reads its own output of the shared text encoder), and the mask."""
        enc = self.tokenizer.encode(prompt)
        ids = np.array([enc.ids], dtype=np.int64)
        att = np.array([enc.attention_mask], dtype=np.int64)
        wanted = ["last_hidden_state"] + ([self.names_output] if "jang" in self.paths else [])
        outs = self.text.run(wanted, {"input_ids": ids, "attention_mask": att})
        return dict(zip(("mo", "jang"), (o.astype(np.float32) for o in outs))), att.astype(bool)

    def route(self, text):
        """Which network suits the prompt: returns (route, network of each candidate as a function of its index)."""
        if "jang" not in self.paths:
            return "single", lambda k: "mo"
        key = alias_key(text, self.aliases)
        # Exactly a name the model was trained on: Jang knows it. (Also when the prompt already spells out the full name.)
        if key is not None and (key == text or self.aliases[key][0] in text):
            return "everyday_object", lambda k: "jang"
        if text in self.known:
            return "known_item", lambda k: "jang"
        # Anything else is a name with extra words, or a combination nobody drew before. The relay is the best single
        # choice there, but Mo follows a color word better and Jang keeps a known item better, so one of each is mixed in.
        return "mixed", lambda k: MIXED[k % len(MIXED)]

    def generate(self, prompt, count, seed, steps, guidance, style, use_expand, init=None, strength=1.0, network="auto"):
        """init is an optional 16x16 RGBA draft; strength says how far the model may move away from it (1 = ignore it).
        network is auto, mo, jang or relay. Returns (sprites, prompt sent to the model, style, network of each sprite, route)."""
        text = prompt.strip()
        if self.manifest["text"].get("lowercase", True):
            text = text.lower()
        expanded, known_style = expand(text, self.aliases) if use_expand else (text, None)
        if style is None:
            style = known_style if known_style is not None else self.manifest["defaults"].get("source", 0.0)
        route, pick = self.route(text)
        if network != "auto" and "jang" in self.paths:
            route, pick = "forced", lambda k: network
        networks = [pick(k) for k in range(count)]
        ctxs, mask = self.encode(expanded)
        nulls, null_mask = self.encode("")
        size = 16
        total = self.manifest["diffusion"]["timesteps"]
        x = np.random.default_rng(seed).standard_normal((count, 4, size, size)).astype(np.float32)
        ts = np.round(np.linspace(total - 1, 0, steps)).astype(np.int64)
        first = 0
        if init is not None and strength < 1:
            # Start part-way through sampling from a noised copy of the draft instead of from pure noise.
            first = min(steps - 1, int(round((1 - strength) * steps)))
            draft = init.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
            draft[:3, init[..., 3] == 0] = -1.0
            a0 = self.alphas[ts[first]]
            x = (np.sqrt(a0) * draft[None] + np.sqrt(1 - a0) * x).astype(np.float32)
        # Every candidate keeps its own noise, so candidate k is the same image whatever the count or the mix.
        for name in sorted(set(networks)):
            rows = [k for k, n in enumerate(networks) if n == name]
            x[rows] = self.sample(name, x[rows], ctxs, mask, nulls, null_mask, ts, first, guidance, style, size)
        rgba = np.clip((x.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).round().astype(np.uint8)
        opaque = rgba[..., 3] >= 128
        rgba[..., 3] = np.where(opaque, 255, 0)
        rgba[~opaque] = 0
        return list(rgba), expanded, (float(style) if self.has_style else None), networks, route

    def sample(self, plan, x, ctxs, mask, nulls, null_mask, ts, first, guidance, style, size):
        """plan is mo, jang or relay (Mo for the first RELAY share of the steps, then Jang)."""
        count, steps = len(x), len(ts)
        feed = {"mask": np.concatenate([np.repeat(mask, count, 0), np.repeat(null_mask, count, 0)]),
                "size_idx": np.full(2 * count, self.manifest["sizes"][str(size)], dtype=np.int64)}
        if self.has_style:
            feed["src_mix"] = np.full(2 * count, style, dtype=np.float32)
        ctx = {n: np.concatenate([np.repeat(ctxs[n], count, 0), np.repeat(nulls[n], count, 0)]) for n in ctxs}
        handoff = round(RELAY * steps)
        for i in range(first, steps):
            t = ts[i]
            name = plan if plan != "relay" else "mo" if i < handoff else "jang"
            v = self.unet(name).run(None, {"x": np.concatenate([x, x]), "t": np.full(2 * count, t, dtype=np.int64),
                                           "ctx": ctx[name], **feed})[0]
            v = v[count:] + guidance * (v[:count] - v[count:])
            a, s = np.sqrt(self.alphas[t]), np.sqrt(1 - self.alphas[t])
            x0 = np.clip(a * x - s * v, -1, 1)
            if i == steps - 1:
                return x0.astype(np.float32)
            eps = (x - a * x0) / s
            nxt = self.alphas[ts[i + 1]]
            x = (np.sqrt(nxt) * x0 + np.sqrt(1 - nxt) * eps).astype(np.float32)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Generate 16x16 Minecraft-style item sprites with the StarItem SIC model.")
    ap.add_argument("prompts", nargs="*", help="item names or short descriptions, Chinese or English")
    ap.add_argument("--out", default="sprites", help="output folder (default: sprites)")
    ap.add_argument("--models", help="folder with the model files")
    ap.add_argument("--count", type=int, default=4, help="sprites per prompt (default 4)")
    ap.add_argument("--seed", type=int, help="fixed seed; the same seed and prompt give the same sprites")
    ap.add_argument("--steps", type=int, help="sampling steps (model default 30)")
    ap.add_argument("--guidance", type=float, help="how closely to follow the prompt (model default 5.5)")
    ap.add_argument("--style", default="auto",
                    help="auto, or a number from 0 (hand-drawn mod sprite look) to 1 (look of the everyday-object sprites)")
    ap.add_argument("--no-expand", action="store_true", help="send the prompt to the model exactly as written")
    ap.add_argument("--network", choices=["auto", "mo", "jang", "relay"], default="auto",
                    help="which network draws (models from SIC-1.2 on have two): auto picks per prompt, jang knows items by "
                         "name, mo follows colors and new combinations more readily, relay lets Mo start and Jang finish")
    ap.add_argument("--ref", metavar="IMAGE",
                    help="picture or photo of one object; the sprites follow its shape and colors")
    ap.add_argument("--ref-strength", type=float, default=0.65,
                    help="0 to 1: how far the model may move away from the reference (default 0.65; 1 ignores it)")
    ap.add_argument("--ref-colors", type=float, default=0.7,
                    help="0 to 1: how strongly the sprites take over the reference's colors (default 0.7; 0 leaves them as generated)")
    ap.add_argument("--scale", type=int, help="enlargement of the preview image (default: 16, less when there are many sprites)")
    ap.add_argument("--preview-of", nargs="+", metavar="PNG",
                    help="make one preview image from existing sprite files, to compare candidates from different runs")
    ap.add_argument("--search", metavar="TEXT", help="list item names the model knows that contain TEXT, then exit")
    args = ap.parse_args()

    def scale_for(n):
        return max(1, args.scale) if args.scale else max(4, min(16, 1600 // (n * 18 + 2)))

    def shown(path):
        return Path(path).resolve().as_posix()

    if args.preview_of:
        sprites = [read_png(f) for f in args.preview_of]
        if len({s.shape for s in sprites}) != 1:
            fail("the files given to --preview-of must all have the same size")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sheet = out / "compare_preview.png"
        write_png(sheet, preview(sprites, scale_for(len(sprites))))
        print(json.dumps({"preview": shown(sheet), "order": [shown(f) for f in args.preview_of]}, ensure_ascii=False))
        return

    folder = find_models(args.models)
    if args.search is not None:
        aliases = json.loads((folder / "aliases.json").read_text(encoding="utf-8")) if (folder / "aliases.json").is_file() else {}
        q = args.search.strip().lower()
        hits = sorted((k for k in aliases if q in k), key=lambda k: (len(k), k))[:60]
        coverage = load_coverage(folder)
        print(json.dumps({"query": args.search, "known_names": [
            {"name": k, "full_name": aliases[k][0], "training_sprites": coverage.get(aliases[k][0])} for k in hits]},
            ensure_ascii=False))
        return
    if not args.prompts:
        ap.error("give at least one prompt, or use --search")
    if not 1 <= args.count <= 16:
        ap.error("--count must be between 1 and 16")
    style = None if args.style == "auto" else float(args.style)

    if not 0 <= args.ref_strength <= 1 or not 0 <= args.ref_colors <= 1:
        ap.error("--ref-strength and --ref-colors must be between 0 and 1")
    ref_info = {}
    draft = load_reference(args.ref, matting=folder / "matting.onnx", info=ref_info) if args.ref else None

    model = Model(folder)
    d = model.manifest["defaults"]
    steps, guidance = args.steps or d["steps"], args.guidance if args.guidance is not None else d["guidance"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, prompt in enumerate(args.prompts):
        seed = (args.seed + n) if args.seed is not None else int.from_bytes(os.urandom(4), "little")
        sprites, expanded, used_style, networks, route = model.generate(
            prompt, args.count, seed, steps, guidance, style, not args.no_expand, init=draft, strength=args.ref_strength,
            network=args.network)
        if draft is not None:
            sprites = [keep_colors(sprite, draft, args.ref_colors) for sprite in sprites]
        slug = re.sub(r"[^\w]+", "_", prompt.strip().lower()).strip("_")[:40] or "sprite"
        files, warnings = [], []
        for k, sprite in enumerate(sprites):
            path = out / f"{slug}_{seed}_{k + 1}.png"
            write_png(path, sprite)
            files.append(shown(path))
            covered = int((sprite[..., 3] > 0).sum())
            if covered < 24:
                warnings.append(f"candidate {k + 1} is nearly empty ({covered} of 256 pixels drawn)")
            elif covered > 250:
                warnings.append(f"candidate {k + 1} has no transparent background")
        sheet = out / f"{slug}_{seed}_preview.png"
        # With a reference, its 16x16 draft comes first in the preview so the candidates can be compared against it.
        tiles = ([draft] if draft is not None else []) + sprites
        write_png(sheet, preview(tiles, scale_for(len(tiles))))
        known = full_name_in(expanded, model.coverage)
        result = {"prompt": prompt, "sent_to_model": expanded, "name_completed": expanded != prompt.strip().lower(),
                  "training_sprites": model.coverage.get(known) if known else None,
                  "style": used_style, "seed": seed, "model": model.manifest.get("model_version"),
                  "files": files, "preview": shown(sheet), "warnings": warnings}
        if route != "single":
            result.update({"route": route, "networks": networks})
        if draft is not None:
            ref_file = out / f"{slug}_{seed}_ref.png"
            write_png(ref_file, draft)
            result.update({"reference": shown(args.ref), "ref_strength": args.ref_strength, "ref_colors": args.ref_colors,
                           "reference_draft": shown(ref_file), "reference_background": ref_info.get("background"),
                           "preview_note": "the first tile is the reference shrunk to 16x16, the rest are the candidates"})
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
