"""从已下载的 jar 中提取物品贴图、lang 和物品模型，并生成贴图索引。

Reads downloads/manifest.json, writes dataset/raw/<project_id>_<slug>/{textures,lang,models}/<namespace>/...
and dataset/raw/textures_index.csv. Each mod's output directory is rebuilt on every run.

Usage:
  python -X utf8 tools/extract_assets.py
  python -X utf8 tools/extract_assets.py --langs all
  python -X utf8 tools/extract_assets.py --only 223852,398521
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import re
import shutil
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "downloads" / "manifest.json"
JARS = ROOT / "downloads" / "jars"

ASSET = re.compile(r"^assets/([^/]+)/(textures/items?|lang|models/item)/(.+)$", re.I)
# 原版命名空间下的贴图是对 Mojang 素材的覆盖，不收
SKIP_NAMESPACES = {"minecraft", "realms"}
# 许可复核中确认不可用的文件（相对 assets/<namespace>/ 的路径）
# 目前为空：BetterNether 的船贴图是 CC BY-NC-SA 4.0，确认不商用后不再剔除
EXCLUDE = {}
WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
INDEX_FIELDS = ["project_id", "slug", "namespace", "source_path", "file", "width", "height", "frames", "has_mcmeta", "sha1"]


def safe_parts(rel):
    """把 zip 内路径转成可以落盘的路径片段；不安全的路径返回 None。"""
    parts = []
    for part in PurePosixPath(rel).parts:
        if part in ("", ".", "..") or "\\" in part or ":" in part:
            return None
        part = re.sub(r'[<>"|?*\x00-\x1f]', "_", part).rstrip(". ") or "_"
        if part.split(".")[0].lower() in WIN_RESERVED:
            part = "_" + part
        parts.append(part)
    return parts or None


def png_size(data):
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return None, None


def find_jar(rec):
    p = ROOT / rec.get("path", "")
    if rec.get("path") and p.is_file():
        return p
    found = sorted((JARS / f"{rec['project_id']}_{rec['slug']}").glob("*.jar"))
    return found[0] if found else None


def extract(rec, out_root, langs):
    pid, slug = rec["project_id"], rec["slug"]
    stat = Counter()
    rows = []
    jar = find_jar(rec)
    if not jar:
        return "no_jar", stat, rows
    dest_root = out_root / f"{pid}_{slug}"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    excludes = EXCLUDE.get(pid, [])
    try:
        z = zipfile.ZipFile(jar)
    except zipfile.BadZipFile:
        return "bad_zip", stat, rows
    with z:
        names = {i.filename for i in z.infolist()}
        used = set()
        # 只读外层 jar 的条目：内嵌的依赖 jar（jar-in-jar）是别的模组，不展开
        for info in z.infolist():
            m = ASSET.match(info.filename)
            if not m or info.is_dir():
                continue
            ns, kind, rel = m.group(1), m.group(2).lower(), m.group(3)
            if ns.lower() in SKIP_NAMESPACES:
                stat["skipped_namespace"] += 1
                continue
            low = rel.lower()
            if kind.startswith("textures"):
                group = "textures"
                if not (low.endswith(".png") or low.endswith(".png.mcmeta")):
                    continue
            elif kind == "lang":
                group = "lang"
                code, _, ext = low.rpartition(".")
                if ext not in ("json", "lang") or "/" in rel:
                    continue
                if langs and code not in langs:
                    continue
                rel = low  # 旧版是 en_US.lang，统一成小写
            else:
                group = "models"
                if not low.endswith(".json"):
                    continue
            inner = f"{kind}/{rel}"
            if any(fnmatch.fnmatch(inner.lower(), pat) for pat in excludes):
                stat["excluded_by_license"] += 1
                continue
            parts = safe_parts(rel)
            if not parts:
                stat["unsafe_path"] += 1
                continue
            dest = dest_root.joinpath(group, *safe_parts(ns), *parts)
            # Windows 文件系统不区分大小写，zip 里只差大小写的两个文件会互相覆盖
            if str(dest).lower() in used:
                dest = dest.with_name(f"{dest.stem}~{info.CRC:08x}{dest.suffix}")
            used.add(str(dest).lower())
            data = z.read(info)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if group == "textures" and low.endswith(".png"):
                w, h = png_size(data)
                if w is None:
                    stat["not_png"] += 1
                    dest.unlink()
                    continue
                has_meta = info.filename + ".mcmeta" in names
                # 动画贴图是竖向排列的帧条
                frames = h // w if w and h > w and h % w == 0 else 1
                rows.append({
                    "project_id": pid, "slug": slug, "namespace": ns, "source_path": info.filename,
                    "file": dest.relative_to(out_root).as_posix(), "width": w, "height": h,
                    "frames": frames, "has_mcmeta": int(has_meta), "sha1": hashlib.sha1(data).hexdigest(),
                })
                stat["textures"] += 1
            elif group == "textures":
                stat["mcmeta"] += 1
            else:
                stat[group] += 1
    return "ok", stat, rows


def main():
    ap = argparse.ArgumentParser(description="Extract item textures, lang files and item models from mod jars.")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "raw", help="output directory")
    ap.add_argument("--langs", default="en_us,zh_cn", help="comma-separated language codes, or 'all'")
    ap.add_argument("--only", help="comma-separated project ids to process")
    args = ap.parse_args()

    langs = None if args.langs == "all" else {c.strip().lower() for c in args.langs.split(",") if c.strip()}
    recs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        recs = [r for r in recs if r["project_id"] in want]
    if not recs:
        sys.exit("Nothing to do.")
    args.out.mkdir(parents=True, exist_ok=True)

    index_path = args.out / "textures_index.csv"
    done = {r["project_id"] for r in recs}
    index = []
    if args.only and index_path.exists():
        # 部分运行时保留其他模组已有的索引行
        with index_path.open(encoding="utf-8-sig", newline="") as f:
            index = [r for r in csv.DictReader(f) if int(r["project_id"]) not in done]

    total = Counter()
    problems = []
    for i, rec in enumerate(sorted(recs, key=lambda r: r["project_id"]), 1):
        status, stat, rows = extract(rec, args.out, langs)
        index.extend(rows)
        total.update(stat)
        note = " ".join(f"{k}={v}" for k, v in sorted(stat.items()))
        print(f"[{i:3}/{len(recs)}] {status:8} {rec['name'][:30]:30} {note}", flush=True)
        if status != "ok" or not stat["textures"] or not stat["lang"]:
            problems.append(f"{rec['project_id']} {rec['name']}: {status} textures={stat['textures']} lang={stat['lang']}")

    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(index)

    sizes = Counter()
    for r in index:
        w_, h_ = int(r["width"]), int(r["height"])
        if (w_, h_) == (16, 16):
            sizes["16x16"] += 1
        elif w_ == 16 and int(r["frames"]) > 1:
            sizes["16px animated strip"] += 1
        elif w_ == h_:
            sizes[f"{w_}x{h_}"] += 1
        else:
            sizes["other"] += 1
    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
    print("Index rows:", len(index), "| unique by sha1:", len({r["sha1"] for r in index}))
    print("Sizes:", ", ".join(f"{k}={v}" for k, v in sizes.most_common(8)))
    print(f"Index: {index_path}")
    for p in problems:
        print("CHECK", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
