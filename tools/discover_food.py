"""第三批候选（食物专项）：农夫乐事、森罗物语厨房的附属和其他食物模组。

Searches Modrinth by the food category and by add-on keywords, looks at each hit's dependencies to find the
parent mods, maps the hits to CurseForge projects (the downloader works through CurseForge) and audits the repos.

Stages (each cached under license_audit/cache):
  python -X utf8 tools/discover_food.py discover          # Modrinth search + dependencies + CurseForge id lookup
  python -X utf8 tools/discover_food.py audit <clone_dir> # repo license files + item texture counts
  python -X utf8 tools/discover_food.py audit-mr <clone_dir>  # the same for mods that are only on Modrinth
"""
import csv
import difflib
import json
import re
import sys
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_repo_licenses as A  # noqa: E402
import discover_batch2 as D  # noqa: E402
import resolve_repos as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "license_audit" / "food_candidates.json"
AUDITED = ROOT / "license_audit" / "food_audit.json"
SKIPPED = ROOT / "license_audit" / "food_skipped.json"
AUDITED_MR = ROOT / "license_audit" / "food_audit_mr.json"
# 只在 Modrinth 上的模组没有 CurseForge 项目号，用它的 Modrinth id 换算一个不会和 CurseForge 撞的整数
MR_BASE = 10 ** 15
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

QUERIES = ["delight", "farmer's delight", "farmersdelight", "kaleidoscope", "cookery", "kitchen", "cuisine", "food", "cooking",
           "森罗物语", "农夫乐事"]
KEYWORD = re.compile(r"(?i)delight|farmer|kaleidoscope|cookery|kitchen|cuisine|food|cook|bak(e|ing)|brew|drink|meal|dish|snack|"
                     r"dessert|candy|fruit|crop|乐事|厨|料理|食")
# 两个主模组在 Modrinth 上的 slug
PARENTS = {"farmers-delight", "farmers-delight-refabricated", "kaleidoscope-cookery"}
MIN_MR_DOWNLOADS = 3_000
MIN_CF_DOWNLOADS = 5_000
PAGES = 10


def search(query, facets):
    hits = []
    for offset in range(0, PAGES * 100, 100):
        params = {"facets": json.dumps(facets), "index": "downloads", "limit": 100, "offset": offset}
        if query:
            params["query"] = query
        _, d = D.get_json("https://api.modrinth.com/v2/search?" + urllib.parse.urlencode(params))
        if not d or not d["hits"]:
            break
        hits += d["hits"]
        if d["hits"][-1]["downloads"] < MIN_MR_DOWNLOADS:
            break
    return hits


def dependencies(project_id):
    """这个模组各版本声明过的依赖项目（slug 和许可证）。"""
    f = R.CACHE / f"mr_deps_{project_id}.json"
    if not f.exists():
        code, d = D.get_json(f"https://api.modrinth.com/v2/project/{project_id}/dependencies")
        if d is None:
            return []
        f.write_text(json.dumps([{"slug": p["slug"], "title": p["title"], "license": (p.get("license") or {}).get("id", "")}
                                 for p in d.get("projects", [])], ensure_ascii=False), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


CF_LINK = re.compile(r"curseforge\.com/minecraft/mc-mods/([\w-]+)")
CF_BADGE = re.compile(r"cf\.way2muchnoise\.eu/(?:full_|short_|title/|versions/)?(\d{4,8})")


LOADER = re.compile(r"(?i)[\[(][^\])]*[\])]|\b(neo)?forge\b|\bfabric\b|\bquilt\b|\ba farmer'?s delight add-?on\b")


# 两边 slug 对不上、靠猜也猜不到的，手工指定
CF_SLUGS = {"kaleidoscopecookery-automation": ["kaleidoscope-cookery-automation"]}


def same_mod(a, b):
    """两个平台上的名字是不是同一个模组：去掉 [Forge]、(Fabric) 这类后缀再比。"""
    a, b = R.norm(LOADER.sub(" ", a)), R.norm(LOADER.sub(" ", b))
    return bool(a and b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85)


def map_to_cf(c):
    bare = LOADER.sub(" ", c["name"])
    slugs = [c["mr_slug"], re.sub(r"[^a-z0-9]+", "-", c["name"].lower()).strip("-"), re.sub(r"[^a-z0-9]+", "-", bare.lower()).strip("-")]
    slugs += [f"{slugs[-1]}-forge", re.sub(r"-(forge|fabric|neoforge)$", "", c["mr_slug"])]
    slugs += CF_SLUGS.get(c["mr_slug"], [])
    for slug in dict.fromkeys(slugs):
        if not slug or (R.CACHE / f"cfwidget_slug_{slug}.404").exists():
            continue
        cf = D.cf_lookup(slug)
        if cf and same_mod(cf["cf_name"], c["name"]):
            return {**c, **cf}
    return {**c, "project_id": None}


def cf_by_id(pid):
    f = R.CACHE / f"cfwidget_id_{pid}.json"
    if not f.exists():
        _, d = D.get_json(f"https://api.cfwidget.com/{pid}")
        if not d or "id" not in d:
            return None
        f.write_text(json.dumps(d), encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    slug = (d.get("urls") or {}).get("curseforge", "").rstrip("/").rsplit("/", 1)[-1]
    return D.cf_lookup(slug) if slug else None


def map_by_links(c, body):
    """Modrinth 的 slug 和 CurseForge 对不上时，从 Modrinth 介绍和仓库 README 里找 CurseForge 链接，名字相近才算数。"""
    _, readme = A.get(f"https://raw.githubusercontent.com/{c['repo']}/HEAD/README.md")
    text = (body or "") + "\n" + (readme if readme and not readme.startswith("__ERR__") else "")
    found = []
    for slug in dict.fromkeys(CF_LINK.findall(text)):
        if not (R.CACHE / f"cfwidget_slug_{slug}.404").exists():
            found.append(D.cf_lookup(slug))
    for pid in dict.fromkeys(CF_BADGE.findall(text)):
        found.append(cf_by_id(pid))
    for cf in found:
        if cf and same_mod(cf["cf_name"], c["name"]):
            return {**c, **cf, "mapped_by": "link"}
    return c


def discover():
    have = {r["slug"] for r in csv.DictReader((ROOT / "mods_final.csv").open(encoding="utf-8-sig"))}
    reviewed = {}
    for name in ("asset_license_review.csv", "batch2_review.csv"):
        for r in csv.DictReader((ROOT / "license_audit" / name).open(encoding="utf-8-sig")):
            reviewed[r["slug"]] = r["verdict"]
    hits = {}
    for h in search("", [["project_type:mod"], ["categories:food"]]):
        hits[h["project_id"]] = h
    for q in QUERIES:
        for h in search(q, [["project_type:mod"]]):
            text = f"{h['title']} {h['description']} {' '.join(h.get('categories', []))}"
            if KEYWORD.search(text):
                hits.setdefault(h["project_id"], h)
    hits = [h for h in hits.values() if h["downloads"] >= MIN_MR_DOWNLOADS]
    print("modrinth hits:", len(hits), "| licenses:", Counter(h["license"] for h in hits).most_common(12))

    allowed, skipped = [], []
    for h in hits:
        (allowed if h["license"] in D.LICENSES else skipped).append(h)
    detail = {}
    for i in range(0, len(allowed), 100):
        ids = json.dumps([h["project_id"] for h in allowed[i:i + 100]])
        _, d = D.get_json("https://api.modrinth.com/v2/projects?" + urllib.parse.urlencode({"ids": ids}))
        for p in d or []:
            detail[p["id"]] = p
    with ThreadPoolExecutor(4) as ex:
        deps = dict(zip([h["project_id"] for h in hits], ex.map(dependencies, [h["project_id"] for h in hits])))

    def parents(h):
        return sorted({d["slug"] for d in deps[h["project_id"]]} & PARENTS)

    SKIPPED.write_text(json.dumps(
        [{"mr_slug": h["slug"], "name": h["title"], "mr_downloads": h["downloads"], "mr_license": h["license"], "parents": parents(h)}
         for h in sorted(skipped, key=lambda h: -h["downloads"])], ensure_ascii=False, indent=1), encoding="utf-8")
    cands = []
    for h in allowed:
        p = detail.get(h["project_id"], {})
        m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", p.get("source_url") or "")
        cands.append({"mr_slug": h["slug"], "name": h["title"], "mr_downloads": h["downloads"], "mr_license": h["license"],
                      "repo": f"{m.group(1)}/{re.sub(r'\.git$', '', m.group(2))}" if m else None,
                      "parents": parents(h), "categories": h.get("categories", []),
                      "deps": [d for d in deps[h["project_id"]] if d["slug"] not in PARENTS]})
    print("allowed license:", len(cands), "| with github source:", sum(1 for c in cands if c["repo"]),
          "| skipped for license:", len(skipped), "of which add-ons of the two mods:", sum(1 for h in skipped if parents(h)))
    with ThreadPoolExecutor(4) as ex:
        mapped = list(ex.map(map_to_cf, [c for c in cands if c["repo"]]))
        bodies = {h["slug"]: detail.get(h["project_id"], {}).get("body", "") for h in allowed}
        # 只对两个主模组的附属和食物分类的模组做这一步，其余的关键词命中太杂
        retry = [m for m in mapped if not m["project_id"] and (m["parents"] or "food" in m["categories"])]
        fixed = dict(zip([m["mr_slug"] for m in retry], ex.map(lambda m: map_by_links(m, bodies.get(m["mr_slug"])), retry)))
        mapped = [fixed.get(m["mr_slug"], m) for m in mapped]
        print("mapped through links:", sum(1 for m in fixed.values() if m.get("project_id")), "of", len(retry))
    res = []
    for m in mapped:
        if not m["project_id"]:
            m["status"] = "not on CurseForge under the same name"
        elif m["cf_slug"] in have:
            continue
        elif m["cf_slug"] in reviewed:
            m["status"] = f"reviewed before: {reviewed[m['cf_slug']]}"
        elif m["cf_downloads"] < MIN_CF_DOWNLOADS:
            m["status"] = "few downloads"
        else:
            m["status"] = "new"
        res.append(m)
    res.sort(key=lambda r: -(r.get("cf_downloads") or 0))
    seen = set()
    res = [r for r in res if not r["project_id"] or not (r["project_id"] in seen or seen.add(r["project_id"]))]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("status:", dict(Counter(r["status"].split(":")[0] for r in res)))
    print("new add-ons of the two mods:", sum(1 for r in res if r["status"] == "new" and r["parents"]),
          "| other new food mods:", sum(1 for r in res if r["status"] == "new" and not r["parents"]))


def audit(clones):
    cands = [c for c in json.loads(OUT.read_text(encoding="utf-8")) if c["status"] == "new"]
    clones.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: D.audit_one(c, clones), cands))
    AUDITED.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("audited:", len(res), "repo ok:", sum(r["ok"] for r in res), "with tree:", sum(r["items"] is not None for r in res))


def mr_project_id(mr_id):
    n = 0
    for ch in mr_id:
        n = n * 62 + BASE62.index(ch)
    return MR_BASE + n


def mr_details(c):
    """补上 Modrinth 项目 id 和作者，字段名沿用 CurseForge 那一套，后面的脚本不用分两种写法。"""
    f = R.CACHE / f"mr_project_{c['mr_slug']}.json"
    if not f.exists():
        _, d = D.get_json(f"https://api.modrinth.com/v2/project/{urllib.parse.quote(c['mr_slug'])}")
        _, team = D.get_json(f"https://api.modrinth.com/v2/project/{urllib.parse.quote(c['mr_slug'])}/members")
        if not d:
            return None
        owner = next((m["user"]["username"] for m in team or [] if m.get("role") == "Owner"), "")
        owner = owner or next((m["user"]["username"] for m in team or []), "")
        f.write_text(json.dumps({"id": d["id"], "slug": d["slug"], "author": owner}), encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    return {**c, "project_id": mr_project_id(d["id"]), "mr_id": d["id"], "cf_slug": d["slug"], "cf_name": c["name"],
            "cf_license": "", "author": d["author"], "cf_downloads": c["mr_downloads"], "modrinth_only": True}


def audit_mr(clones):
    cands = [c for c in json.loads(OUT.read_text(encoding="utf-8"))
             if c["status"].startswith("not on") and (c["parents"] or "food" in c["categories"])]
    clones.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        cands = [c for c in ex.map(mr_details, cands) if c]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: D.audit_one(c, clones), cands))
    AUDITED_MR.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("audited:", len(res), "repo ok:", sum(r["ok"] for r in res), "with tree:", sum(r["items"] is not None for r in res))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "discover":
        discover()
    elif len(sys.argv) >= 3 and sys.argv[1] == "audit":
        audit(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "audit-mr":
        audit_mr(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
