"""为清单中的每个模组解析源码仓库地址：cfwidget 描述里的链接优先，Modrinth 搜索兜底。"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "curseforge_permissive_mods.csv"
CACHE = ROOT / "license_audit" / "cache"
OUT = ROOT / "license_audit" / "repos.json"
UA = {"User-Agent": "StarItem-license-audit/0.1"}
REPO_RE = re.compile(r"https?://(github\.com|gitlab\.com)/([\w.-]+)/([\w.-]+)", re.I)
# 描述里常见的非本体仓库
SKIP_OWNERS = {"sponsors", "orgs", "users", "topics", "apps", "marketplace", "intergrav"}


def get(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    return f"__ERR__ {err}"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def cf_candidates(pid):
    f = CACHE / f"cfwidget_{pid}.json"
    if not f.exists():
        f.write_text(get(f"https://api.cfwidget.com/{pid}"), encoding="utf-8")
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    found = []
    for host, owner, repo in REPO_RE.findall(d.get("description") or ""):
        if owner.lower() in SKIP_OWNERS:
            continue
        found.append(f"https://{host.lower()}/{owner}/{re.sub(r'\.git$', '', repo)}")
    return [u for u, _ in Counter(found).most_common()]


def modrinth_candidate(name):
    f = CACHE / f"modrinth_{norm(name)}.json"
    if not f.exists():
        q = urllib.parse.quote(name)
        f.write_text(get(f"https://api.modrinth.com/v2/search?query={q}&limit=5"), encoding="utf-8")
    try:
        hits = json.loads(f.read_text(encoding="utf-8")).get("hits", [])
    except json.JSONDecodeError:
        return None
    for h in hits:
        # 只接受标题规范化后完全一致的结果，避免错配到附属模组
        if norm(h["title"]) == norm(name):
            pf = CACHE / f"modrinth_project_{h['slug']}.json"
            if not pf.exists():
                pf.write_text(get(f"https://api.modrinth.com/v2/project/{h['slug']}"), encoding="utf-8")
            try:
                p = json.loads(pf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            return {"slug": h["slug"], "source_url": p.get("source_url"), "license": (p.get("license") or {}).get("id")}
    return None


def resolve(row):
    cands = cf_candidates(row["project_id"])
    mr = modrinth_candidate(row["name"])
    return {
        "project_id": row["project_id"],
        "slug": row["slug"],
        "name": row["name"],
        "cf_license": row["license"],
        "cf_repos": cands[:4],
        "modrinth": mr,
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(resolve, rows))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for i, r in enumerate(res, 1):
        mr = r["modrinth"] or {}
        print(f"{i:3} {r['name'][:28]:28} cf={(r['cf_repos'] or ['-'])[0][19:60]:42} mr={(mr.get('source_url') or '-')[19:60]}")


if __name__ == "__main__":
    sys.exit(main())
