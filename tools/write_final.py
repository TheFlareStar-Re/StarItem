"""合并出最终清单 mods_final.csv：复核通过的 84 个 + 从候选里补位的 16 个。"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "license_audit" / "asset_license_review.csv"
SRC = ROOT / "curseforge_permissive_mods.csv"
FILL = ROOT / "license_audit" / "fill_candidates.json"
OUT = ROOT / "mods_final.csv"
KEEP = {"ok_explicit", "ok_default", "ok_weak", "partial"}

# 补位：仓库有宽松许可文件、无素材单独条款、物品贴图不少于 5 张，按下载量取前 16
FILL_SLUGS = """ender-storage-1-8 compact-machines rftools-power mining-gadgets toms-storage laserio
dynamictrees regions-unexplored advanced-generators fairy-lights scaling-health more-red
fabric-waystones jaopca chickens project-red-fabrication""".split()
# 多模块仓库：只算本模块目录下的物品贴图
ITEM_COUNT_FIX = {"project-red-fabrication": 12}


def main():
    src = {r["project_id"]: r for r in csv.DictReader(SRC.open(encoding="utf-8-sig"))}
    out = []
    for r in csv.DictReader(REVIEW.open(encoding="utf-8-sig")):
        if r["verdict"] in KEEP:
            s = src[r["project_id"]]
            out.append({**{k: s[k] for k in ("project_id", "slug", "name", "author", "downloads")},
                        "cf_license": r["cf_license"], "repo": r["repo"], "verdict": r["verdict"],
                        "item_textures_in_repo": "", "note": r["note"], "batch": "original"})
    fill = {e["slug"]: e for e in json.loads(FILL.read_text(encoding="utf-8"))}
    for slug in FILL_SLUGS:
        e = fill[slug]
        lic = sorted({k for d, v in e["docs"].items() if "readme" not in d.lower() and "fml" not in d.lower()
                      and "forge" not in d.lower() for k in v["kinds"]})
        assert e["project_id"] and lic and e["items"] >= 5, (slug, e["project_id"], lic, e["items"])
        out.append({"project_id": e["project_id"], "slug": slug, "name": e["name"], "author": e["author"],
                    "downloads": int(round(e["downloads_m"] * 1e6)), "cf_license": e["cf_license"],
                    "repo": f"https://github.com/{e['repo']}", "verdict": "ok_default",
                    "item_textures_in_repo": ITEM_COUNT_FIX.get(slug, e["items"]),
                    "note": f"仓库许可文件：{'/'.join(lic)}，未见素材单独条款", "batch": "fill"})
    out.sort(key=lambda r: -int(r["downloads"]))
    assert len(out) == 100 and len({r["project_id"] for r in out}) == 100, len(out)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), dict(Counter(r["verdict"] for r in out)), dict(Counter(r["batch"] for r in out)))
    print("min downloads:", out[-1]["name"], out[-1]["downloads"])
    for r in out:
        if r["batch"] == "fill":
            print(f"{r['project_id']:>7} {r['name'][:30]:30} {r['author'][:18]:18} {int(r['downloads'])/1e6:6.1f}M {r['cf_license'][:8]:8} items={r['item_textures_in_repo']:<4} {r['repo'][19:]}")


if __name__ == "__main__":
    sys.exit(main())
