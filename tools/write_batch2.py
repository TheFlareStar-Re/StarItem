"""把第二批的人工复核结论写入 license_audit/batch2_review.csv，并把入选的追加到 mods_final.csv。"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDITED = ROOT / "license_audit" / "batch2_audit.json"
REVIEW = ROOT / "license_audit" / "batch2_review.csv"
FINAL = ROOT / "mods_final.csv"
MIN_ITEMS = 10

DEFAULT = "仓库许可文件：{lic}，未见素材单独条款"
SELECT = {
    # 非商用许可（用户已明确不商用）
    "quark": ("ok_nc", "LICENSE：整个模组 CC BY-NC-SA 3.0"),
    "applied-energistics-2": ("ok_nc", "README：贴图和模型 CC BY-NC-SA 3.0（代码 LGPL）"),
    "create-big-cannons": ("ok_nc", "LICENSE：手绘贴图和模型 CC BY-NC-SA 4.0"),
    "armourers-workshop": ("ok_nc", "LICENSE：CC BY-NC-SA 3.0"),
    "biomancy": ("ok_nc", "LICENSE：美术素材 CC BY-NC 4.0（代码 MIT）"),
    "kaleidoscope-cookery": ("ok_nc", "LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0"),
    "kaleidoscope-tavern": ("ok_nc", "LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0"),
    "oritech": ("ok_nc", "LICENSE 为 CC0；README：部分素材来自 techarium（CC BY-NC 4.0）和 malcolmriley（CC BY 4.0），需署名"),
    # 素材被明确置于宽松许可下
    "blood-magic": ("ok_explicit", "LICENSE：整个项目 CC BY 4.0，需署名"),
    "medieval-origins-revival": ("ok_explicit", "LICENSE：素材 CC BY 4.0，需署名"),
    "tinkers-levelling-addon": ("ok_explicit", "README：代码、贴图、二进制均为 MIT"),
    "cyclic": ("ok_default", "LICENSE 为 MIT；item/charm 目录的 CREDIT.txt 提到可能用了 CC BY-SA 3.0 的素材"),
}
SELECT_DEFAULT = """actually-additions construction-sticks mariums-soulslike-weaponry chalk chalk-fabric cognition-mod
building-wands habitat rustic-delight armor-of-the-ages tarot-cards backpacks dawn-of-time numismatic-overhaul
ubes-delight collectors-reap veggies-delight amethyst-imbuement dungeonz trail-tales-delight
jadens-nether-expansion-delight exquisito eden-ring-reforked kitchen-karrot cosmopolitan beans-backpacks
braziliandelight immortalers-delight""".split()

REJECT = {
    "素材保留所有权利": "apotheosis mythicmetals fish-of-thieves easy-npc easy-npc-core paradise-lost medievalweapons "
                "clockwork create-fabric cobblecuisine rainbow-reef aquaculture",
    "GPL/LGPL": "genshin-instruments casualness-delight create-enchantment-industry-fabric-legacy",
    "依附于素材许可更严的模组": "zenith create-deco create-framed create-sound-of-steam create-propulsion-simulated "
                    "power-grid create-fly create-diesel-generators-fabric ad-astra-giselle-addon thaumon "
                    "valkyrien-skies-tournament",
    "移植版或与已有模组重复": "farmers-delight-refabricated farmers-delight-fabric cultural-delights-fabric "
                   "corn-delight-fabric dumplings-delight-reload bloodmagic",
    "素材可能来自 Mojang 或其他游戏": "copper-age-backport stardew-fishing station-decoration goety",
    "物品目录里不是物品图标": "natures-compass explorers-compass from-the-fog",
}


def repo_license(e):
    skip = ("readme", "fml", "forge", "template", "third", "credit")
    return sorted({k for fn, v in e["docs"].items() if not any(t in fn.lower() for t in skip) for k in v["kinds"]})


def main():
    audited = json.loads(AUDITED.read_text(encoding="utf-8"))
    reject = {slug: why for why, slugs in REJECT.items() for slug in slugs.split()}
    existing = [r for r in csv.DictReader(FINAL.open(encoding="utf-8-sig")) if r["batch"] != "batch2"]
    have_ids = {r["project_id"] for r in existing}
    have_repos = {r["repo"].lower() for r in existing}

    review, added = [], []
    for e in sorted(audited, key=lambda e: -e["cf_downloads"]):
        slug = e["cf_slug"]
        # 截断显示时用过的名字和真实 slug 对不上的情况
        key = next((k for k in list(SELECT) + SELECT_DEFAULT if slug == k), None)
        lic = "/".join(repo_license(e))
        repo = f"https://github.com/{e['repo']}"
        if str(e["project_id"]) in have_ids or repo.lower() in have_repos:
            verdict, note = "skip", "已在清单中（同一项目或同一仓库）"
        elif key in SELECT:
            verdict, note = SELECT[key]
        elif key:
            verdict, note = "ok_default", DEFAULT.format(lic=lic)
        elif slug in reject:
            verdict, note = "exclude", reject[slug]
        elif (e["items"] or 0) < MIN_ITEMS:
            verdict, note = "skip", f"仓库里物品贴图不足 {MIN_ITEMS} 张"
        elif not lic:
            verdict, note = "unverified", "仓库无许可文件"
        else:
            verdict, note = "unreviewed", ""
        row = {"project_id": e["project_id"], "slug": slug, "name": e["name"], "author": e["author"],
               "downloads": e["cf_downloads"], "cf_license": e["cf_license"] or f"Modrinth: {e['mr_license'] or '-'}",
               "repo": repo, "verdict": verdict, "item_textures_in_repo": e["items"] if e["items"] is not None else "",
               "note": note, "batch": "batch2"}
        review.append(row)
        if verdict.startswith("ok"):
            assert lic or verdict != "ok_default", slug
            added.append(row)

    missing = [k for k in list(SELECT) + SELECT_DEFAULT if k not in {r["slug"] for r in added}]
    fields = list(existing[0])
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(review)
    with FINAL.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + added)
    print("review:", dict(Counter(r["verdict"] for r in review)))
    print("unreviewed:", [r["slug"] for r in review if r["verdict"] == "unreviewed"])
    print("selected but not added:", missing)
    print("added:", len(added), dict(Counter(r["verdict"] for r in added)),
          "| item textures in repos:", sum(int(r["item_textures_in_repo"] or 0) for r in added))
    print("mods_final.csv rows:", len(existing) + len(added))


if __name__ == "__main__":
    sys.exit(main())
