"""把第三批（食物专项）的人工复核结论写入 license_audit/food_review.csv，并把入选的追加到 mods_final.csv。

Run tools/discover_food.py (discover, audit, audit-mr) first. Re-running this script rewrites the "food" and
"food_mr" rows only. "food_mr" rows are downloaded from Modrinth: their slug is the Modrinth slug, and mods that
exist only there get a project id above 10**15 derived from their Modrinth id.

Usage:
  python -X utf8 tools/write_batch3.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_batch2 import DEFAULT, repo_license  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUDITED = ROOT / "license_audit" / "food_audit.json"
AUDITED_MR = ROOT / "license_audit" / "food_audit_mr.json"
REVIEW = ROOT / "license_audit" / "food_review.csv"
FINAL = ROOT / "mods_final.csv"
BATCH = "food"
BATCH_MR = "food_mr"
# CurseForge 上作者关闭了第三方下载、接口不给地址的，改从 Modrinth 下
MR_FALLBACK = {"christmas-delight", "foodables-fabric", "eternal-starlight-delight", "spoiledz"}
MIN_ITEMS = 10

SELECT = {
    # 非商用许可
    "kaleidoscope-grilling": ("ok_nc", "LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0"),
    "bakeries": ("ok_nc", "LICENSE-ASSETS.txt：资源文件 CC BY-NC-SA 4.0"),
    "butchercraft": ("ok_nc", "LICENSE.md：整个项目 CC BY-NC-SA 4.0"),
    "legume-delight": ("ok_nc", "LICENSE.txt：整个项目 CC BY-NC-SA 4.0"),
    "kaleidoscope-dimensions-wine": ("ok_nc", "LICENSE.md：assets 目录 CC BY-NC-SA 4.0（森罗物语酒馆的附属）"),
    # 素材被明确置于宽松许可下
    "paper-kite-manor": ("ok_explicit", "LICENSE-ASSETS：资源文件 MIT"),
    "light-meals": ("ok_explicit", "仓库根目录和 resources 目录下的 LICENSE 均为 MIT"),
    "dumplings-delight-rewrapped": ("ok_default", "LICENSE：MIT；移植自 FoggyHillside 的 Dumplings Delight（原作在 Modrinth 声明 MIT）"),
    # 以下几个只在 Modrinth 上
    "sakura_mod": ("ok_default", "LICENSE：MIT；resources 里附带的是农夫乐事的 MIT 许可"),
    "vegehenna": ("ok_explicit", "LICENSE：整个项目 CC BY 4.0，需署名"),
    "angling": ("ok_explicit", "LICENSE：CC0"),
    "farmers-crops": ("ok_explicit", "LICENSE：CC0"),
    "farmers-expanse": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "large-meals": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "spanish-delight": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "veggie-delight": ("ok_weak", "仓库无许可文件；fabric.mod.json 声明 CC BY 4.0，Modrinth 页面声明 MIT"),
    # 仓库里没有许可文件，只有构建文件里的声明；这一个声明的是非商用许可，按非商用归类
    "extradelight": ("ok_nc", "仓库无许可文件；gradle.properties、mods.toml 和 Modrinth 页面均声明 CC BY-NC-SA 4.0（非商用）"),
    "hearth-and-harvest": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "critters-delight": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "peruvians-delight": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "yeast-n-feast": ("ok_weak", "仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT"),
    "hotkettles": ("ok_weak", "仓库无许可文件；fabric.mod.json、mods.toml 和 Modrinth 页面声明 MIT"),
}
SELECT_DEFAULT = """hibernal-herbs modern-delight frights-delight realisticdining arbitrary-delight eternal-starlight-delight
ramadan-delight vanilla-delight avaritias-delight more-delight-forge christmas-delight barbeques-delight corn-delight
easters-delight copper-pot maple-delight fungi-delight
usefulfood-reborn kawaii-dishes foodables-fabric enhanced-farming raven-coffee bodacious-berries valleymans-beans
candy-world-recaramelized super-apple simply-tea tea-aroma spoiledz extended-mushrooms
muffins-thais-delight photosynthesis bagels-baking tasty-supplies gone-fishing fishing-101
millet-delight farmers-delight-plus crafty-cuisine nears farmers-delight-expansion vietnams-delight-reborn""".split()

REJECT = {
    "素材保留所有权利": "mooncake-delight calypsos-candy-workshop bettervannilafishes food-pluss teyvats-delight "
                "el-and-ls-dye-mod assorted-discoveries",
    "GPL/LGPL": "frycooks-delight better-snowball-fight",
    "依附于素材许可更严的模组": "vampires-delight lenderscataclysm-delight hybrid-delights respiteful cake-delight "
                    "delightful-creators-forge respite-creators-fabric pizza-delight garnished garnished-additions "
                    "create-garnished-reworked create-blazing-hot croptopia-additions kaleidoscope-bloodwine "
                    "starcatchers-delight compat-o-plenty no-mans-delight fintastic peculiar-primordials "
                    "some-assembly-required farmers-delight-extended irregular-chef enderscapes-delight abundant-delight "
                    "firmalife createaddition creates-delight tough-as-nails-create-addon wcsm borukva-food-exotic",
    "移植版或与已有模组重复": "kaleidoscope-cookery-refabricated kaleidoscope-tavern-refabricated kaleidoscope-end-refabricated "
                   "rusticdelight-refabricated croptopia-refabricated oceans-delight-polymer-port "
                   "spanish-delight-refabricated kaleidoscope-nether-refabricated",
    "素材可能来自 Mojang 或其他游戏": "delta-delight stardew-fishing-fabric kamen-rider-craft umapyoi umapyoi-delight "
                          "umapyoi-fabric matcha-flavoured mystias_izakaya touhoupixelcanteen unicopia",
    "数据包，不是 jar 模组": "kaleidoscope-deco magicalfoods",
    "素材可能来自许可未核实的原作": "food-expansion-reimagined",
}
# mooncake-delight、assorted-discoveries：许可文件写素材可用，README 却写美术素材保留所有权利，按更严的算


def main():
    audited = json.loads(AUDITED.read_text(encoding="utf-8"))
    if AUDITED_MR.exists():
        audited += json.loads(AUDITED_MR.read_text(encoding="utf-8"))
    reject = {slug: why for why, slugs in REJECT.items() for slug in slugs.split()}
    existing = [r for r in csv.DictReader(FINAL.open(encoding="utf-8-sig")) if r["batch"] not in (BATCH, BATCH_MR)]
    have_ids = {r["project_id"] for r in existing}
    have_repos = {r["repo"].lower() for r in existing}

    review, added = [], []
    for e in sorted(audited, key=lambda e: -e["cf_downloads"]):
        slug = e["cf_slug"]
        lic = "/".join(repo_license(e)) if e["ok"] else ""
        repo = f"https://github.com/{e['repo']}"
        parents = "、".join(dict.fromkeys({"farmers-delight": "农夫乐事", "farmers-delight-refabricated": "农夫乐事",
                                          "kaleidoscope-cookery": "森罗物语厨房"}[p] for p in e["parents"]))
        if str(e["project_id"]) in have_ids or repo.lower() in have_repos:
            verdict, note = "skip", "已在清单中（同一项目或同一仓库）"
        elif slug in SELECT:
            verdict, note = SELECT[slug]
        elif slug in SELECT_DEFAULT:
            verdict, note = "ok_default", DEFAULT.format(lic=lic)
        elif slug in reject:
            verdict, note = "exclude", reject[slug]
        elif not e["ok"]:
            verdict, note = "unverified", "仓库打不开"
        elif (e["items"] or 0) < MIN_ITEMS:
            verdict, note = "skip", f"仓库里物品贴图不足 {MIN_ITEMS} 张"
        elif not lic:
            verdict, note = "unverified", "仓库无许可文件"
        else:
            verdict, note = "skip", "不是食物类，本批不收"
        if parents and verdict != "skip":
            note += f"；{parents}的附属"
        row = {"project_id": e["project_id"], "slug": slug, "name": e["name"], "author": e["author"],
               "downloads": e["cf_downloads"], "cf_license": e["cf_license"] or f"Modrinth: {e['mr_license'] or '-'}",
               "repo": repo, "verdict": verdict, "item_textures_in_repo": e["items"] if e["items"] is not None else "",
               "note": note, "batch": BATCH_MR if e.get("modrinth_only") else BATCH}
        if slug in MR_FALLBACK and verdict.startswith("ok"):
            row.update({"slug": e["mr_slug"], "batch": BATCH_MR,
                        "note": note + "；CurseForge 关闭了第三方下载，从 Modrinth 下载"})
        review.append(row)
        if verdict.startswith("ok"):
            # 没有许可文件的只能是"证据较弱"；extradelight 例外，它的构建文件声明的是非商用许可
            assert lic or verdict == "ok_weak" or slug == "extradelight", slug
            added.append(row)

    got = {r["slug"] for r in added} | (MR_FALLBACK if any(r["batch"] == BATCH_MR for r in added) else set())
    missing = [k for k in list(SELECT) + SELECT_DEFAULT if k not in got]
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
    print("selected but not added:", missing)
    print("added:", len(added), dict(Counter(r["verdict"] for r in added)),
          "| item textures in repos:", sum(int(r["item_textures_in_repo"] or 0) for r in added))
    print("mods_final.csv rows:", len(existing) + len(added))
    for batch in (BATCH, BATCH_MR):
        rows = [r for r in added if r["batch"] == batch]
        print(f"batch {batch}: {len(rows)} mods, {sum(int(r['item_textures_in_repo'] or 0) for r in rows)} item textures in repos")


if __name__ == "__main__":
    sys.exit(main())
