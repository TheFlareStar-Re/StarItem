"""抓取各仓库根目录的 LICENSE/README/CREDITS 类文件，找出与贴图素材许可相关的段落。"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = ROOT / "license_audit" / "repos.json"
FILES = ROOT / "license_audit" / "files"
OUT = ROOT / "license_audit" / "audit_raw.json"
UA = {"User-Agent": "StarItem-license-audit/0.1"}

# 两个接口都没给出仓库或给错的，手工指定
OVERRIDES = {
    "Natura": "SlimeKnights/Natura",
    "AE2 Things [Forge]": "Technici4n/AE2Things-Forge",
    "Mystical World": "MysticMods/MysticalWorld",
    "Just Dire Things": "Direwolf20-MC/JustDireThings",
    "Advanced Rocketry": "zmaster587/AdvancedRocketry",
    "Food Expansion": "Lellson/FoodExpansion",
    "Eidolon": "elucent/eidolon",
    "Better Survival mod": "mujmajnkraft/BetterSurvival",
    "Things [Fabric]": "wisp-forest/things",
    "Dank Storage": "Tfarcenim/Dank-Storage",
}
WANT = re.compile(r"(?i)licen[sc]e|copying|notice|credit|readme|unlicense|asset|attribution|legal")
TOPIC = re.compile(r"(?i)\b(assets?|textures?|art(work)?s?|sprites?|models?|sounds?|music|graphics|images?)\b")
TERMS = re.compile(
    r"(?i)licen[sc]|all rights reserved|\bARR\b|copyright|creative commons|\bCC[- ]?(BY|0)|permission|"
    r"not (be )?(covered|licensed|used|redistribut)|may not|do not|except|property of|belongs? to|credit"
)
KINDS = [
    ("MIT", r"Permission is hereby granted, free of charge|\bMIT License\b"),
    ("Apache-2.0", r"Apache License\s+Version 2\.0"),
    ("LGPL", r"GNU LESSER GENERAL PUBLIC LICENSE"),
    ("GPL", r"GNU GENERAL PUBLIC LICENSE"),
    ("MPL-2.0", r"Mozilla Public License"),
    ("CC0", r"CC0 1\.0|Creative Commons Zero"),
    ("CC-BY-NC", r"Attribution-NonCommercial|CC[- ]BY[- ]NC"),
    ("CC-BY", r"Creative Commons Attribution|CC[- ]BY"),
    ("Unlicense", r"This is free and unencumbered software"),
    ("WTFPL", r"DO WHAT THE FUCK YOU WANT"),
    ("BSD", r"Redistribution and use in source and binary forms"),
    ("ARR", r"(?i)all rights reserved"),
]


def get(url, retries=3):
    err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.geturl(), r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return url, None
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(3 * (i + 1))
    return url, f"__ERR__ {err}"


def pick_repo(r):
    if r["name"] in OVERRIDES:
        return OVERRIDES[r["name"]], "manual"
    mr = (r.get("modrinth") or {}).get("source_url") or ""
    m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", mr)
    if m:
        return f"{m.group(1)}/{m.group(2)}", "modrinth"
    for u in r["cf_repos"]:
        m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", u)
        if m and m.group(1) != "user-attachments":
            return f"{m.group(1)}/{m.group(2)}", "cf-description"
    return None, None


def audit(repo):
    d = FILES / repo.replace("/", "__")
    meta_f = d / "_meta.json"
    if meta_f.exists():
        return json.loads(meta_f.read_text(encoding="utf-8"))
    d.mkdir(parents=True, exist_ok=True)
    final, html = get(f"https://github.com/{repo}")
    meta = {"repo": repo, "ok": False, "files": {}}
    if not html or html.startswith("__ERR__"):
        meta["error"] = html or "404"
        return meta  # 失败不落盘，方便重跑
    m = re.search(r"github\.com/([\w.-]+/[\w.-]+)", final)
    meta["resolved"] = m.group(1) if m else repo
    names = sorted(set(re.findall(r'"name":"([^"]+)","path":"\1","contentType":"file"', html)))
    meta["root_files"] = names
    meta["archived"] = "This repository has been archived" in html or '"isArchived":true' in html
    wanted = [n for n in names if WANT.search(n)]
    if not names:
        wanted = ["LICENSE", "LICENSE.md", "LICENSE.txt", "README.md", "COPYING", "CREDITS.md"]
    for n in wanted:
        _, body = get(f"https://raw.githubusercontent.com/{meta['resolved']}/HEAD/{n}")
        if body and not body.startswith("__ERR__"):
            (d / n.replace("/", "_")).write_text(body, encoding="utf-8")
            meta["files"][n] = len(body)
    meta["ok"] = True
    meta_f.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def classify(text):
    head = text[:3000]
    return [k for k, pat in KINDS if re.search(pat, head)]


def snippets(text, limit=6):
    out = []
    for para in re.split(r"\n\s*\n", text):
        p = " ".join(para.split())
        if 20 < len(p) and TOPIC.search(p) and TERMS.search(p):
            out.append(p[:420])
            if len(out) >= limit:
                break
    return out


def main():
    rows = json.loads(REPOS.read_text(encoding="utf-8"))
    picks = {}
    for r in rows:
        repo, how = pick_repo(r)
        picks[r["name"]] = (repo, how)
    uniq = sorted({v[0] for v in picks.values() if v[0]})
    with ThreadPoolExecutor(4) as ex:
        metas = dict(zip(uniq, ex.map(audit, uniq)))
    result = []
    for r in rows:
        repo, how = picks[r["name"]]
        meta = metas.get(repo, {"ok": False, "error": "no repo", "files": {}})
        entry = {
            "project_id": r["project_id"], "name": r["name"], "cf_license": r["cf_license"],
            "repo": meta.get("resolved", repo), "repo_source": how, "ok": meta["ok"],
            "error": meta.get("error"), "archived": meta.get("archived"), "docs": {},
        }
        for fn in meta["files"]:
            text = (FILES / repo.replace("/", "__") / fn.replace("/", "_")).read_text(encoding="utf-8")
            is_readme = "readme" in fn.lower()
            entry["docs"][fn] = {
                "kinds": [] if is_readme else classify(text),
                "snippets": snippets(text),
            }
        result.append(entry)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [e["name"] for e in result if not e["ok"]]
    print(f"repos={len(uniq)} ok={sum(e['ok'] for e in result)}/{len(result)} failed={bad}")


if __name__ == "__main__":
    sys.exit(main())
