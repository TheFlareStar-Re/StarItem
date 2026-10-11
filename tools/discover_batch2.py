"""第二批候选：从 Modrinth 按许可证和分类找内容类模组，映射到 CurseForge 项目，再做仓库许可复核。

Stages (each cached under license_audit/cache):
  python -X utf8 tools/discover_batch2.py discover          # Modrinth search + CurseForge id lookup
  python -X utf8 tools/discover_batch2.py audit <clone_dir> # repo license files + item texture counts
"""
import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_repo_licenses as A  # noqa: E402
import resolve_repos as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "license_audit" / "batch2_candidates.json"
AUDITED = ROOT / "license_audit" / "batch2_audit.json"

# 用户明确不商用，非商用（NC）许可可以接受；禁止演绎（ND）的仍然不收
LICENSES = ["MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "BSD-2-Clause", "BSD-3-Clause", "Unlicense",
            "ISC", "Zlib", "WTFPL", "0BSD", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-3.0", "CC-BY-NC-SA-3.0"]
CATEGORIES = ["food", "equipment", "magic", "technology", "adventure", "mobs"]
TOP_N = 500
MIN_CF_DOWNLOADS = 100_000
# Modrinth 上搜不到或许可证字段不在上面范围内、但仓库许可已知可用的
SEEDS = [
    ("applied-energistics-2", "AppliedEnergistics/Applied-Energistics-2"),
    ("quark", "VazkiiMods/Quark"),
    ("create-big-cannons", "Cannoneers-of-Create/CreateBigCannons"),
    ("ae2-things-forge", "Technici4n/AE2Things-Forge"),
    ("cognition-mod", "terrarium-earth/Cognition"),
    ("the-spice-of-life", "squeek502/SpiceOfLife"),
    ("bloodmagic", "WayofTime/BloodMagic"),
    ("cyclic", "Lothrazar/Cyclic"),
    ("actually-additions", "Ellpeck/ActuallyAdditions"),
    ("aquaculture", "TeamMetallurgy/Aquaculture"),
    ("apotheosis", "Shadows-of-Fire/Apotheosis"),
]
ITEM_TEX = re.compile(r"(?i)assets/[^/]+/textures/items?/.+\.png$")
NESTED = re.compile(r"(?i)(^|/)(licen[sc]e|copying|credits?|notice|attribution)[^/]*$")


def get_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=R.UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return e.code, None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (i + 1))
    return 0, None


def modrinth_search():
    facets = json.dumps([["project_type:mod"], [f"license:{x}" for x in LICENSES], [f"categories:{c}" for c in CATEGORIES]])
    hits = []
    for offset in range(0, TOP_N, 100):
        q = urllib.parse.urlencode({"facets": facets, "index": "downloads", "limit": 100, "offset": offset})
        _, d = get_json("https://api.modrinth.com/v2/search?" + q)
        if not d or not d["hits"]:
            break
        hits += d["hits"]
    # 详情里才有 source_url
    detail = {}
    for i in range(0, len(hits), 100):
        ids = json.dumps([h["project_id"] for h in hits[i:i + 100]])
        _, d = get_json("https://api.modrinth.com/v2/projects?" + urllib.parse.urlencode({"ids": ids}))
        for p in d or []:
            detail[p["id"]] = p
    out = []
    for h in hits:
        p = detail.get(h["project_id"], {})
        m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", p.get("source_url") or "")
        out.append({"mr_slug": h["slug"], "name": h["title"], "mr_downloads": h["downloads"], "mr_license": h["license"],
                    "repo": f"{m.group(1)}/{re.sub(r'\.git$', '', m.group(2))}" if m else None})
    return out


def cf_lookup(slug):
    """用 cfwidget 按 slug 查 CurseForge 项目；查不到返回 None。"""
    f = R.CACHE / f"cfwidget_slug_{slug}.json"
    if not f.exists():
        code, d = get_json(f"https://api.cfwidget.com/minecraft/mc-mods/{slug}")
        if not d or "id" not in d:
            if code == 404:
                f.with_suffix(".404").write_text("", encoding="utf-8")
            return None
        f.write_text(json.dumps(d), encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    members = d.get("members") or []
    return {"project_id": d["id"], "cf_slug": slug, "cf_name": d.get("title", ""), "cf_license": d.get("license", ""),
            "author": members[0]["username"] if members else "", "cf_downloads": (d.get("downloads") or {}).get("total", 0)}


def map_to_cf(c):
    title_slug = re.sub(r"[^a-z0-9]+", "-", c["name"].lower()).strip("-")
    for slug in dict.fromkeys([c["mr_slug"], title_slug]):
        if (R.CACHE / f"cfwidget_slug_{slug}.404").exists():
            continue
        cf = cf_lookup(slug)
        # slug 相同不代表是同一个模组，名称规范化后也要一致
        if cf and R.norm(cf["cf_name"]) == R.norm(c["name"]):
            return {**c, **cf}
    return {**c, "project_id": None}


def discover():
    have = {r["slug"] for r in csv.DictReader((ROOT / "curseforge_permissive_mods.csv").open(encoding="utf-8-sig"))}
    have |= {r["slug"] for r in csv.DictReader((ROOT / "mods_final.csv").open(encoding="utf-8-sig"))}
    seeds_slugs = {s for s, _ in SEEDS}
    have -= seeds_slugs  # 之前因非商用条款被排除的，现在可以重新进入
    cands = [c for c in modrinth_search() if c["repo"]]
    print("modrinth hits with github source:", len(cands))
    with ThreadPoolExecutor(4) as ex:
        mapped = list(ex.map(map_to_cf, cands))
    res = [m for m in mapped if m["project_id"] and m["cf_slug"] not in have and m["cf_slug"] not in seeds_slugs
           and m["cf_downloads"] >= MIN_CF_DOWNLOADS]
    for slug, repo in SEEDS:
        cf = cf_lookup(slug)
        if cf:
            res.append({"mr_slug": None, "name": cf["cf_name"], "mr_downloads": 0, "mr_license": "", "repo": repo, **cf, "seed": True})
        else:
            print("seed not resolved:", slug)
    seen = set()
    res = [r for r in sorted(res, key=lambda r: -r["cf_downloads"]) if not (r["project_id"] in seen or seen.add(r["project_id"]))]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("mapped to CurseForge:", sum(1 for m in mapped if m["project_id"]), "| new with >=100k CF downloads:", len(res))


def tree(repo, clones):
    f = R.CACHE / f"tree_{repo.replace('/', '__')}.txt"
    if not f.exists():
        d = clones / repo.replace("/", "__")
        if not d.exists():
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                                f"https://github.com/{repo}", str(d)], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return None
        r = subprocess.run(["git", "-C", str(d), "-c", "core.quotepath=off", "ls-tree", "-r", "--name-only", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode != 0:
            return None
        f.write_text(r.stdout, encoding="utf-8")
    return f.read_text(encoding="utf-8").splitlines()


def audit_one(c, clones):
    e = dict(c, docs={}, items=None, nested=[], ok=False)
    try:
        meta = A.audit(c["repo"])
    except Exception as ex:  # noqa: BLE001
        e["error"] = str(ex)
        return e
    e["ok"] = meta["ok"]
    if not meta["ok"]:
        return e
    e["repo"] = meta.get("resolved", c["repo"])
    for fn in meta["files"]:
        text = (A.FILES / c["repo"].replace("/", "__") / fn.replace("/", "_")).read_text(encoding="utf-8")
        e["docs"][fn] = {"kinds": [] if "readme" in fn.lower() else A.classify(text), "snippets": A.snippets(text, 4)}
    paths = tree(e["repo"], clones)
    if paths is not None:
        e["items"] = sum(1 for p in paths if ITEM_TEX.search(p))
        e["nested"] = [p for p in paths if "/" in p and NESTED.search(p) and "/lang/" not in p][:6]
    return e


def audit(clones):
    cands = json.loads(OUT.read_text(encoding="utf-8"))
    clones.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: audit_one(c, clones), cands))
    AUDITED.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("audited:", len(res), "repo ok:", sum(r["ok"] for r in res), "with tree:", sum(r["items"] is not None for r in res))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "discover":
        discover()
    elif len(sys.argv) >= 3 and sys.argv[1] == "audit":
        audit(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
