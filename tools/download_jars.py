"""下载 mods_final.csv 中每个模组的最新 jar，并记录来源清单。

Most mods come from the CurseForge API, which needs a key: set env CURSEFORGE_API_KEY, or put the key on the
first line of cf_api_key.txt in the project root. Rows whose batch ends with "_mr" are only on Modrinth and
come from the Modrinth API, which needs no key.

Usage:
  python -X utf8 tools/download_jars.py                  # latest file of every mod
  python -X utf8 tools/download_jars.py --game-version 1.20.1
  python -X utf8 tools/download_jars.py --only 223852,398521 --dry-run
  python -X utf8 tools/download_jars.py --batch food     # only the mods added in one batch
  python -X utf8 tools/download_jars.py --batch food_mr  # Modrinth-only mods, no API key needed
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.curseforge.com/v1"
MODRINTH = "https://api.modrinth.com/v2"
RELEASE_RANK = {"release": 1, "beta": 2, "alpha": 3}
KEY_FILE = ROOT / "cf_api_key.txt"
UA = "StarItem-downloader/0.1"
# 附带发布的源码包、开发包不是我们要的
NOT_MAIN = re.compile(r"(?i)[-_.](sources?|src|dev|deobf|api|javadoc|slim)\.jar$")
HASH_SHA1 = 1


def load_key():
    key = os.environ.get("CURSEFORGE_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        lines = KEY_FILE.read_text(encoding="utf-8-sig").split()
        key = lines[0] if lines else ""
    if not key:
        sys.exit(f"No API key. Set CURSEFORGE_API_KEY or write the key into {KEY_FILE}")
    return key


def request(url, key=None, retries=5):
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    err = None
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(f"HTTP {e.code} (API key rejected or no access): {url}") from e
            if e.code == 404:
                raise RuntimeError(f"HTTP 404: {url}") from e
            err = e  # 429 和 5xx 退避后重试
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"{err}: {url}")


def api_get(path, key, **params):
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with request(url, key) as r:
        return json.load(r)


def pick_file(files):
    """正式版优先于 beta/alpha，同级取最新。"""
    ok = [
        f for f in files
        if f.get("isAvailable", True)
        and f["fileName"].lower().endswith(".jar")
        and not NOT_MAIN.search(f["fileName"])
        and not f.get("isServerPack")
        and not f.get("parentProjectFileId")
    ]
    if not ok:
        return None
    best_type = min(f.get("releaseType", 3) for f in ok)
    return max((f for f in ok if f.get("releaseType", 3) == best_type), key=lambda f: f["fileDate"])


def sha1_of(path):
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    tmp = dest.with_name(dest.name + ".part")
    with request(url) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def is_modrinth(row):
    return row.get("batch", "").endswith("_mr")


def process_modrinth(row, args):
    """Modrinth 上的模组：slug 一栏是 Modrinth 的 slug，正式版优先，同级取最新。"""
    pid = row["project_id"]
    rec = {"project_id": int(pid), "slug": row["slug"], "name": row["name"], "status": "error", "source": "modrinth"}
    try:
        with request(f"{MODRINTH}/project/{urllib.parse.quote(row['slug'])}/version") as r:
            versions = json.load(r)
        if args.game_version:
            matched = [v for v in versions if args.game_version in v.get("game_versions", [])]
            rec["matched_game_version"] = bool(matched)
            versions = matched or versions
        ok = []
        for v in versions:
            files = [f for f in v.get("files", []) if f["filename"].lower().endswith(".jar") and not NOT_MAIN.search(f["filename"])]
            if files:
                ok.append((v, next((f for f in files if f.get("primary")), files[0])))
        if not ok:
            rec["status"] = "no_file"
            return rec
        best = min(RELEASE_RANK.get(v["version_type"], 3) for v, _ in ok)
        v, f = max(((v, f) for v, f in ok if RELEASE_RANK.get(v["version_type"], 3) == best), key=lambda x: x[0]["date_published"])
        sha1 = (f.get("hashes") or {}).get("sha1")
        rec.update({
            "file_id": v["id"], "file_name": f["filename"], "file_date": v["date_published"], "file_length": f.get("size"),
            "release_type": RELEASE_RANK.get(v["version_type"], 3), "game_versions": v.get("game_versions", []) + v.get("loaders", []),
            "sha1": sha1, "download_url": f["url"],
        })
        dest = args.out / f"{pid}_{row['slug']}" / f["filename"]
        rec["path"] = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest)
        if args.dry_run:
            rec["status"] = "dry_run"
            return rec
        if dest.exists() and sha1 and sha1_of(dest) == sha1:
            rec["status"] = "cached"
            return rec
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(f["url"], dest)
        if sha1 and sha1_of(dest) != sha1:
            dest.unlink()
            rec["status"] = "hash_mismatch"
            return rec
        rec["status"] = "downloaded"
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def process(row, key, args):
    if is_modrinth(row):
        return process_modrinth(row, args)
    pid = row["project_id"]
    rec = {"project_id": int(pid), "slug": row["slug"], "name": row["name"], "status": "error"}
    try:
        files = []
        if args.game_version:
            files = api_get(f"/mods/{pid}/files", key, gameVersion=args.game_version, pageSize=50)["data"]
            rec["matched_game_version"] = bool(files)
        if not files:
            files = api_get(f"/mods/{pid}/files", key, pageSize=50)["data"]
        f = pick_file(files)
        if not f:
            rec["status"] = "no_file"
            return rec
        sha1 = next((h["value"].lower() for h in f.get("hashes", []) if h.get("algo") == HASH_SHA1), None)
        rec.update({
            "file_id": f["id"], "file_name": f["fileName"], "file_date": f["fileDate"],
            "file_length": f.get("fileLength"), "release_type": f.get("releaseType"),
            "game_versions": f.get("gameVersions", []), "sha1": sha1, "download_url": f.get("downloadUrl"),
        })
        dest = args.out / f"{pid}_{row['slug']}" / f["fileName"]
        rec["path"] = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest)
        if not f.get("downloadUrl"):
            # 作者关闭了第三方分发，API 不给下载地址，需要到网页手动下载
            rec["status"] = "no_download_url"
            rec["manual_url"] = f"https://www.curseforge.com/minecraft/mc-mods/{row['slug']}/files/{f['id']}"
            return rec
        if args.dry_run:
            rec["status"] = "dry_run"
            return rec
        if dest.exists() and sha1 and sha1_of(dest) == sha1:
            rec["status"] = "cached"
            return rec
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(f["downloadUrl"], dest)
        if sha1 and sha1_of(dest) != sha1:
            dest.unlink()
            rec["status"] = "hash_mismatch"
            return rec
        rec["status"] = "downloaded"
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Download latest mod jars from CurseForge and Modrinth.")
    ap.add_argument("--list", type=Path, default=ROOT / "mods_final.csv", help="CSV with project_id, slug, name")
    ap.add_argument("--out", type=Path, default=ROOT / "downloads" / "jars", help="output directory")
    ap.add_argument("--game-version", help="prefer files for this Minecraft version, e.g. 1.20.1")
    ap.add_argument("--only", help="comma-separated project ids to process")
    ap.add_argument("--batch", help="only the mods with this value in the list's batch column, e.g. food")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="resolve files without downloading")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.list.open(encoding="utf-8-sig")))
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if r["project_id"] in want]
    if args.batch:
        rows = [r for r in rows if r.get("batch") == args.batch]
    if not rows:
        sys.exit("Nothing to do.")

    # 只有 CurseForge 的模组需要 key；先用一个请求验证，避免 100 个任务全部报同样的错
    key = None
    cf_rows = [r for r in rows if not is_modrinth(r)]
    if cf_rows:
        key = load_key()
        try:
            api_get(f"/mods/{cf_rows[0]['project_id']}", key)
        except RuntimeError as e:
            sys.exit(f"API check failed: {e}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(lambda r: process(r, key, args), rows), 1):
            results.append(rec)
            detail = rec.get("file_name") or rec.get("error") or ""
            print(f"[{i:3}/{len(rows)}] {rec['status']:15} {rec['name'][:32]:32} {detail}", flush=True)

    manifest = args.out.parent / "manifest.json"
    # 部分运行（--only）时保留其他模组已有的记录
    merged = {}
    if manifest.exists():
        merged = {m["project_id"]: m for m in json.loads(manifest.read_text(encoding="utf-8"))}
    if not args.dry_run:
        for r in results:
            old = merged.get(r["project_id"], {})
            # 这次请求失败时，不要用错误记录盖掉之前下载成功的记录
            if r["status"] == "error" and old.get("status") in ("downloaded", "cached"):
                continue
            merged[r["project_id"]] = r
        manifest.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=1), encoding="utf-8")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    manual = [r for r in results if r["status"] == "no_download_url"]
    if manual:
        print("\nDownload these manually (third-party distribution disabled) and save to the path shown:")
        for r in manual:
            print(f"  {r['manual_url']}\n    -> {r['path']}")
    failed = [r for r in results if r["status"] in ("error", "hash_mismatch", "no_file")]
    for r in failed:
        print(f"FAILED {r['project_id']} {r['name']}: {r['status']} {r.get('error', '')}")
    if not args.dry_run:
        print(f"\nManifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
