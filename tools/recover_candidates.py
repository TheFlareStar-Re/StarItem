"""从上一个会话的记录里恢复"已读许可证且为宽松许可"的候选名单。"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = Path("C:/Users/Admin/.claude/projects/D--data-----/bdc543da-a167-4d0a-8ffa-731bc1069971.jsonl")
OUT = ROOT / "license_audit" / "permissive_candidates.json"
KEY = '"rows": '


def result_text(line):
    for p in json.loads(line)["message"]["content"]:
        if p.get("type") == "tool_result":
            c = p["content"]
            return c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c)
    return ""


def main():
    lines = TRANSCRIPT.read_text(encoding="utf-8").splitlines()
    seen = {}
    for i in (129, 145):
        t = result_text(lines[i])
        at = t.index(KEY) + len(KEY)
        s, _ = json.JSONDecoder().raw_decode(t[at:])
        part = [l.split("|") for l in s.split("\n") if l.count("|") == 4]
        print(i, "rows", len(part))
        for path, name, dl, lic, cats in part:
            seen.setdefault(path, {"path": path, "name": name, "downloads_m": float(dl), "license": lic, "categories": cats})
    cand = sorted(seen.values(), key=lambda r: -r["downloads_m"])
    OUT.write_text(json.dumps(cand, ensure_ascii=False, indent=1), encoding="utf-8")
    have = {r["slug"] for r in csv.DictReader((ROOT / "curseforge_permissive_mods.csv").open(encoding="utf-8-sig"))}
    rest = [c for c in cand if c["path"].split("/")[-1] not in have]
    print("total", len(cand), "in list", len(cand) - len(rest), "remaining", len(rest))
    for c in rest:
        print(f"{c['path'].replace('mc-mods/', '')}|{c['name']}|{c['downloads_m']}|{c['license']}|{c['categories']}")


if __name__ == "__main__":
    sys.exit(main())
