"""对补位候选做同样的仓库许可复核，并统计仓库里的物品贴图数量。"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_repo_licenses as A  # noqa: E402
import resolve_repos as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "license_audit" / "permissive_candidates.json"
OUT = ROOT / "license_audit" / "fill_candidates.json"

SLUGS = """xnet macaws-doors compact-machines refined-storage-addons tool-belt rftools-power
wither-skeleton-tweaks mining-gadgets malisisdoors toms-storage laserio advanced-generators
project-red-integration fairy-lights project-red-illumination scaling-health more-red
project-red-expansion illager-invasion fabric-waystones jaopca majruszs-progressive-difficulty
deep-resonance astikorcarts industrial-foregoing-souls chickens project-red-fabrication
metal-barrels cognition-mod regions-unexplored dynamictrees goblin-traders ender-storage-1-8
integrated-terminals colossal-chests the-spice-of-life wireless-redstone-cbe cat-jammies
mystical-agriculture-tiered-crystals angel-ring""".split()
REPO_OVERRIDES = {
    "ender-storage-1-8": "TheCBProject/EnderStorage",
    "wireless-redstone-cbe": "TheCBProject/WR-CBE",
    "tool-belt": "gigaherz/ToolBelt",
    "goblin-traders": "MrCrayfish/GoblinTraders",
    "mining-gadgets": "Direwolf20-MC/MiningGadgets",
    "malisisdoors": "Ordinastie/MalisisDoors",
    "laserio": "Direwolf20-MC/LaserIO",
    "chickens": "setycz/ChickensMod",
    "metal-barrels": "Tfarcenim/MetalBarrels",
    "the-spice-of-life": "squeek502/SpiceOfLife",
    # Modrinth 上同名的是 Fabric 移植版，不是 CurseForge 这个项目
    "fairy-lights": "pau101/Fairy-Lights",
}
CLONES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "license_audit" / "clones"
NESTED = re.compile(r"(?i)(^|/)(licen[sc]e|copying|credits?|notice|attribution)[^/]*$")
ITEM_TEX = re.compile(r"(?i)assets/[^/]+/textures/items?/.+\.png$")


def cf_info(slug):
    f = R.CACHE / f"cfwidget_slug_{slug}.json"
    if not f.exists():
        body = R.get(f"https://api.cfwidget.com/minecraft/mc-mods/{slug}")
        if '"id"' not in body:
            return {}
        f.write_text(body, encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    found = []
    for host, owner, repo in R.REPO_RE.findall(d.get("description") or ""):
        if owner.lower() not in R.SKIP_OWNERS and owner != "user-attachments":
            found.append(f"https://{host.lower()}/{owner}/{re.sub(r'\.git$', '', repo)}")
    members = d.get("members") or []
    return {"id": d.get("id"), "author": members[0]["username"] if members else "", "cf_repos": list(dict.fromkeys(found))[:4]}


def tree(repo):
    # GitHub API 未登录每小时只有 60 次，改用不含文件内容的浅克隆来列目录
    f = R.CACHE / f"tree_{repo.replace('/', '__')}.txt"
    if not f.exists():
        d = CLONES / repo.replace("/", "__")
        if not d.exists():
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                                f"https://github.com/{repo}", str(d)], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                return None
        r = subprocess.run(["git", "-C", str(d), "-c", "core.quotepath=off", "ls-tree", "-r", "--name-only", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode != 0:
            return None
        f.write_text(r.stdout, encoding="utf-8")
    return f.read_text(encoding="utf-8").splitlines()


def one(c):
    slug = c["path"].split("/")[-1]
    info = cf_info(slug)
    row = {"name": c["name"], "cf_repos": info.get("cf_repos", []), "modrinth": R.modrinth_candidate(c["name"])}
    repo, how = (REPO_OVERRIDES[slug], "manual") if slug in REPO_OVERRIDES else A.pick_repo(row)
    e = {"slug": slug, "name": c["name"], "downloads_m": c["downloads_m"], "cf_license": c["license"],
         "categories": c["categories"], "project_id": info.get("id"), "author": info.get("author"),
         "repo": repo, "repo_source": how, "docs": {}, "items": None, "nested": []}
    if not repo:
        return e
    meta = A.audit(repo)
    e["repo"] = meta.get("resolved", repo)
    e["ok"] = meta["ok"]
    for fn in meta["files"]:
        text = (A.FILES / repo.replace("/", "__") / fn.replace("/", "_")).read_text(encoding="utf-8")
        e["docs"][fn] = {"kinds": [] if "readme" in fn.lower() else A.classify(text), "snippets": A.snippets(text)}
    paths = tree(e["repo"])
    if paths is not None:
        e["items"] = sum(1 for p in paths if ITEM_TEX.search(p))
        e["nested"] = [p for p in paths if "/" in p and NESTED.search(p) and "/lang/" not in p][:8]
    return e


def main():
    by_slug = {c["path"].split("/")[-1]: c for c in json.loads(CAND.read_text(encoding="utf-8"))}
    missing = [s for s in SLUGS if s not in by_slug]
    if missing:
        print("not in candidates:", missing)
    todo = [by_slug[s] for s in SLUGS if s in by_slug]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, todo))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for e in sorted(res, key=lambda x: -x["downloads_m"]):
        lic = {k: v["kinds"] for k, v in e["docs"].items() if "readme" not in k.lower()}
        print(f"{e['downloads_m']:6} {e['slug'][:30]:30} id={e['project_id']} {str(e['repo'])[:38]:38} {e['repo_source'] or '-':9} items={e['items']} lic={lic} nested={e['nested']}")
    print()
    for e in res:
        for fn, v in e["docs"].items():
            for s in v["snippets"]:
                print(f"[{e['slug']}] {fn}: {s[:300]}")


if __name__ == "__main__":
    sys.exit(main())
