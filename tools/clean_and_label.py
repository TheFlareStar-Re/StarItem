"""清洗物品贴图并配上名称标注，输出训练用的 RGBA 数据集（16x16 和 32x32 两种分辨率）。

Reads dataset/raw (output of extract_assets.py), writes dataset/clean/images/<slug>/*.png,
dataset/clean/metadata.jsonl and dataset/clean/report.json. dataset/clean is rebuilt on every run.
Requires Pillow and numpy.

Usage:
  python -X utf8 tools/clean_and_label.py
  python -X utf8 tools/clean_and_label.py --no-orphans      # only sprites referenced by an item model
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
MODS = ROOT / "mods_final.csv"
NAME_ZH = ROOT / "dataset" / "name_zh"  # 补充的中文名对照表（*.tsv：name_en、name_zh），给没有中文语言文件的模组用
SIZES = (16, 32)
MIN_OPAQUE = 8  # 16x16 下不透明像素少于这个数的视为空图，32x32 按面积放大

# 没有被任何物品模型引用、且只是"部件图层"的贴图目录（相对 textures/<namespace>/）
PART_LAYERS = {
    "tinkers-construct": ["tool/*"],
    "silent-gear": [
        "arrow/*", "axe/*", "bow/*", "crossbow/*", "dagger/*", "excavator/*", "fishing_rod/*", "hammer/*",
        "hoe/*", "katana/*", "knife/*", "machete/*", "mattock/*", "paxel/*", "pickaxe/*", "prospector_hammer/*",
        "saw/*", "shears/*", "shield/*", "shovel/*", "sickle/*", "slingshot/*", "spear/*", "sword/*",
        "trident/*", "part/*", "elytra/*", "armor/*", "curio/*", "ring/*", "bracelet/*", "necklace/*",
    ],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean_name(s):
    s = re.sub(r"§.", "", s)
    s = re.sub(r"%(\d+\$)?[sdf]", "", s)
    return " ".join(s.split()).strip(" :-")


def humanize(item_id):
    return " ".join(w.capitalize() for w in re.split(r"[_./\s]+", item_id.split("/")[-1]) if w)


# 这些前缀的 key 是物品/方块的显示名，其余（tooltip、gui、advancement 等）不是
NAME_KEY = re.compile(r"^(item|block|tile|gem|material|fluid)\.(.+?)(\.name)?$")
LANG_PAIR = re.compile(r'^\s*("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*")\s*,?\s*$')


def parse_lang_json(text):
    try:
        data = json.loads(text, strict=False)
        return [(k, v) for k, v in data.items() if isinstance(v, str)] if isinstance(data, dict) else []
    except json.JSONDecodeError:
        pass
    # 有些 lang 文件带注释或多余逗号，不是合法 JSON，退回按行取键值对
    pairs = []
    for line in text.splitlines():
        m = LANG_PAIR.match(line)
        if m:
            try:
                pairs.append((json.loads(m.group(1), strict=False), json.loads(m.group(2), strict=False)))
            except json.JSONDecodeError:
                continue
    return pairs


class Lang:
    """一个命名空间下某种语言的名称表，支持新版 json 和旧版 .lang 两种 key 形态。"""

    def __init__(self, ns):
        self.ns = ns
        self.exact = {}
        self.fuzzy = {}

    def load(self, path):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if path.suffix == ".json":
            pairs = parse_lang_json(text)
        else:
            pairs = []
            for line in text.splitlines():
                if "=" in line and not line.lstrip().startswith(("#", "//")):
                    k, v = line.split("=", 1)
                    pairs.append((k.strip(), v.strip()))
        for k, v in pairs:
            v = clean_name(v)
            if not v:
                continue
            self.exact.setdefault(k, v)
            m = NAME_KEY.match(k)
            if not m:
                continue
            rank = 0 if m.group(1) == "item" else 1
            core = m.group(2)
            own = re.sub(rf"^{re.escape(self.ns)}[.:_]", "", core, flags=re.I)
            self._add(norm(own), v, rank)
            # 前缀不是命名空间的情况，如 item.oc.analyzer.name、item.SC2:solar_engine.name
            head, sep, tail = re.split(r"([.:])", core, maxsplit=1) if re.search(r"[.:]", core) else (core, "", "")
            if sep and own == core:
                self._add(norm(tail), v, rank + 2)

    def _add(self, key, value, rank):
        if key and (key not in self.fuzzy or rank < self.fuzzy[key][0]):
            self.fuzzy[key] = (rank, value)

    def lookup(self, item_id):
        dotted = item_id.replace("/", ".")
        last = item_id.split("/")[-1]
        for key in (f"item.{self.ns}.{dotted}", f"item.{self.ns}.{last}", f"block.{self.ns}.{dotted}",
                    f"block.{self.ns}.{last}"):
            if key in self.exact:
                return self.exact[key]
        hit = self.fuzzy.get(norm(item_id)) or self.fuzzy.get(norm(last))
        return hit[1] if hit else None


def load_sprite(path):
    """读成 16 或 32 像素见方的 RGBA；动画帧条取第一帧，只是放大 2 倍的 32x32 还原成 16；其余尺寸返回 None。"""
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:  # noqa: BLE001
        return None, "unreadable"
    w, h = img.size
    animated = False
    if h > w and h % w == 0:
        img = img.crop((0, 0, w, w))
        animated = True
        h = w
    if w != h or w not in SIZES:
        return None, "size"
    a = np.asarray(img)
    if w == 32:
        b = a.reshape(16, 2, 16, 2, 4)
        if (b == b[:, :1, :, :1]).all():
            a = b[:, 0, :, 0]
    return (a.copy(), animated), None


def composite(layers):
    # 图层分辨率不一致时，把 16 的按最近邻放大到 32 再叠
    size = max(layer.shape[0] for layer in layers)
    layers = [layer if layer.shape[0] == size else layer.repeat(2, axis=0).repeat(2, axis=1) for layer in layers]
    out = Image.fromarray(layers[0], "RGBA")
    for layer in layers[1:]:
        out = Image.alpha_composite(out, Image.fromarray(layer, "RGBA"))
    a = np.asarray(out).copy()
    a[a[..., 3] == 0] = 0  # 全透明像素的 RGB 归零，保证内容相同的图哈希一致
    return a


def main():
    ap = argparse.ArgumentParser(description="Clean item sprites and label them with lang names.")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "clean")
    ap.add_argument("--no-orphans", action="store_true", help="drop textures not referenced by any item model")
    args = ap.parse_args()

    mods = {f"{r['project_id']}_{r['slug']}": r for r in csv.DictReader(MODS.open(encoding="utf-8-sig"))}
    mod_dirs = sorted(d for d in RAW.iterdir() if d.is_dir() and d.name in mods)
    drops = Counter()
    extra_zh = {}
    for table in sorted(NAME_ZH.glob("*.tsv")) if NAME_ZH.is_dir() else []:
        for row in csv.DictReader(table.open(encoding="utf-8"), delimiter="	"):
            extra_zh[row["name_en"].strip().lower()] = row["name_zh"].strip()

    # 贴图索引：own[mod][(ns, 相对路径)]，以及跨模组按命名空间查找用的 by_ns
    own = defaultdict(dict)
    by_ns = defaultdict(dict)
    for d in mod_dirs:
        for p in d.glob("textures/*/**/*.png"):
            ns = p.relative_to(d).parts[1]
            rel = p.relative_to(d / "textures" / ns).as_posix()[:-4].lower()
            own[d.name][(ns.lower(), rel)] = p
            by_ns[ns.lower()].setdefault(rel, p)

    samples = []
    for d in mod_dirs:
        info = mods[d.name]
        langs = {}
        for p in sorted(d.glob("lang/*/*")):
            code = p.name.split(".")[0].split("~")[0]
            if code in ("en_us", "zh_cn"):
                langs.setdefault((p.parent.name, code), Lang(p.parent.name)).load(p)

        models = {}
        for p in d.glob("models/*/**/*.json"):
            ns = p.relative_to(d).parts[1]
            item_id = p.relative_to(d / "models" / ns).as_posix()[:-5]
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"), strict=False)
            except json.JSONDecodeError:
                drops["model_bad_json"] += 1
                continue
            if isinstance(data, dict):
                models[(ns, item_id)] = data

        def textures_of(ns, item_id, depth=0):
            """沿 parent 链合并 textures，子模型覆盖父模型。"""
            data = models.get((ns, item_id))
            if data is None or depth > 8:
                return {}
            merged = {}
            parent = data.get("parent")
            if isinstance(parent, str) and ":" in parent:
                pns, ppath = parent.split(":", 1)
                if ppath.startswith("item/"):
                    merged.update(textures_of(pns, ppath[5:], depth + 1))
            t = data.get("textures")
            if isinstance(t, dict):
                merged.update({k: v for k, v in t.items() if isinstance(v, str)})
            return merged

        referenced = set()
        for (ns, item_id) in sorted(models):
            tex = textures_of(ns, item_id)
            layer_keys = sorted((k for k in tex if re.fullmatch(r"layer\d+", k)), key=lambda k: int(k[5:]))
            if not layer_keys:
                drops["model_without_sprite"] += 1
                continue
            paths = []
            for k in layer_keys:
                tns, _, tpath = tex[k].rpartition(":")
                tns = (tns or "minecraft").lower()
                head, _, rest = tpath.partition("/")
                p = None
                if head in ("item", "items") and rest:
                    p = own[d.name].get((tns, rest.lower())) or by_ns.get(tns, {}).get(rest.lower())
                if p is None:
                    paths = None
                    break
                paths.append(p)
            if paths is None:
                drops["model_layer_unresolved"] += 1
                continue
            referenced.update(paths)
            samples.append({"mod": d.name, "info": info, "ns": ns, "item_id": item_id, "kind": "model",
                            "paths": paths, "langs": langs})

        if not args.no_orphans:
            patterns = PART_LAYERS.get(info["slug"], [])
            for (ns, rel), p in sorted(own[d.name].items()):
                if p in referenced:
                    continue
                if any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                    drops["orphan_part_layer"] += 1
                    continue
                samples.append({"mod": d.name, "info": info, "ns": p.relative_to(d).parts[1], "item_id": rel,
                                "kind": "texture", "paths": [p], "langs": langs})
        else:
            drops["orphan_skipped"] += sum(1 for p in own[d.name].values() if p not in referenced)

    # 渲染、过滤、起名
    rendered = []
    for s in samples:
        layers, animated, reason = [], False, None
        for p in s["paths"]:
            got, reason = load_sprite(p)
            if got is None:
                break
            layers.append(got[0])
            animated = animated or got[1]
        if reason:
            drops[reason] += 1
            continue
        img = composite(layers)
        if int((img[..., 3] > 0).sum()) < MIN_OPAQUE * (img.shape[0] // 16) ** 2:
            drops["empty"] += 1
            continue
        en = s["langs"].get((s["ns"], "en_us"))
        zh = s["langs"].get((s["ns"], "zh_cn"))
        name_en = en.lookup(s["item_id"]) if en else None
        name_zh = zh.lookup(s["item_id"]) if zh else None
        shown = name_en or humanize(s["item_id"])
        zh_source = "lang" if name_zh else None
        # 模组没带中文名，或者中文语言文件里原样抄了英文：查补充的对照表
        if not name_zh or name_zh.strip().lower() == shown.strip().lower():
            extra = extra_zh.get(shown.strip().lower())
            if extra:
                name_zh, zh_source = extra, "translated"
        s.update(img=img, animated=animated, name_en=shown, name_zh=name_zh, name_zh_source=zh_source,
                 name_source="lang" if name_en else "id", digest=hashlib.sha1(img.tobytes()).hexdigest())
        rendered.append(s)

    # 像素完全相同的只留一张：有 lang 名称的优先，其次是由物品模型得到的
    rendered.sort(key=lambda s: (s["name_source"] != "lang", s["kind"] != "model", s["mod"], s["item_id"]))
    kept = {}
    for s in rendered:
        first = kept.get(s["digest"])
        if first is None:
            s["alt_names_en"] = []
            kept[s["digest"]] = s
        else:
            drops["duplicate"] += 1
            if s["name_source"] == "lang" and s["name_en"] != first["name_en"] \
                    and s["name_en"] not in first["alt_names_en"] and len(first["alt_names_en"]) < 8:
                first["alt_names_en"].append(s["name_en"])

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    final = sorted(kept.values(), key=lambda s: (s["info"]["slug"], s["ns"], s["item_id"]))
    used = set()
    with (args.out / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as meta:
        for s in final:
            slug = s["info"]["slug"]
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", s["item_id"].replace("/", "__"))
            rel = f"images/{slug}/{stem}.png"
            if rel.lower() in used:
                rel = f"images/{slug}/{stem}~{s['digest'][:8]}.png"
            used.add(rel.lower())
            dest = args.out / rel
            dest.parent.mkdir(exist_ok=True)
            Image.fromarray(s["img"], "RGBA").save(dest, optimize=True)
            meta.write(json.dumps({
                "file": rel, "name_en": s["name_en"], "name_zh": s["name_zh"], "name_source": s["name_source"],
                "name_zh_source": s["name_zh_source"],
                "alt_names_en": s["alt_names_en"], "mod": s["info"]["name"], "slug": slug,
                "project_id": int(s["info"]["project_id"]), "namespace": s["ns"], "item_id": s["item_id"],
                "kind": s["kind"], "layers": [p.relative_to(RAW).as_posix() for p in s["paths"]],
                "size": int(s["img"].shape[0]), "animated": s["animated"], "license_verdict": s["info"]["verdict"],
                "cf_license": s["info"]["cf_license"], "sha1": s["digest"],
            }, ensure_ascii=False) + "\n")

    per_mod = Counter(s["info"]["slug"] for s in final)
    report = {
        "images": len(final),
        "size_16": sum(s["img"].shape[0] == 16 for s in final),
        "size_32": sum(s["img"].shape[0] == 32 for s in final),
        "name_from_lang": sum(s["name_source"] == "lang" for s in final),
        "name_from_id": sum(s["name_source"] == "id" for s in final),
        "with_zh_name": sum(bool(s["name_zh"]) for s in final),
        "zh_name_translated": sum(s["name_zh_source"] == "translated" for s in final),
        "from_model": sum(s["kind"] == "model" for s in final),
        "from_texture_only": sum(s["kind"] == "texture" for s in final),
        "multi_layer": sum(len(s["paths"]) > 1 for s in final),
        "animated_first_frame": sum(s["animated"] for s in final),
        "dropped": dict(sorted(drops.items())),
        "per_mod": dict(per_mod.most_common()),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in report.items():
        if k != "per_mod":
            print(f"{k}: {v}")
    print("mods with images:", len(per_mod), "| top:", per_mod.most_common(6))
    print("mods with none:", sorted(set(m["slug"] for m in mods.values()) - set(per_mod)))
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
