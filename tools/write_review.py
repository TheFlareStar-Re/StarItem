"""把人工复核结论合并进清单，输出 license_audit/asset_license_review.csv。"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "license_audit" / "audit_raw.json"
SRC = ROOT / "curseforge_permissive_mods.csv"
OUT = ROOT / "license_audit" / "asset_license_review.csv"

# 未列出的模组默认 ok_default：仓库根目录有宽松许可文件，且没有针对素材的单独条款
VERDICTS = {
    # 素材被明确置于宽松许可下
    "Tinkers Construct": ("ok_explicit", "README：代码、贴图、二进制均为 MIT"),
    "Integrated Dynamics": ("ok_explicit", "README：代码和图片均为 MIT"),
    "Natura": ("ok_explicit", "LICENSE.md：贴图 CC BY 3.0，需署名"),
    "Tinkers' Mechworks": ("ok_explicit", "README：贴图 CC BY 3.0，需署名"),
    "EvilCraft": ("ok_explicit", "README：代码和图片 CC BY 4.0，需署名（CurseForge 写的是 Public Domain）"),
    "Modern Industrialization": ("ok_explicit", "README：素材全部 CC0；部分贴图改自 GregTech 等他人作品"),
    "OpenComputers": ("ok_explicit", "LICENSE：贴图为 CC0，除非另有标注"),
    "Scannable": ("ok_explicit", "LICENSE：贴图为 CC0，除非另有标注"),
    "Occultism": ("ok_explicit", "THIRD_PARTY_NOTICES：未列出的素材随项目 MIT；列出的第三方图标为 CC BY 3.0，需署名"),
    # 需要剔除部分文件
    "BetterNether": ("partial", "LICENSE.ASSETS：item/*_boat.png、*_chest_boat.png 等为 CC BY-NC-SA 4.0，需剔除；其余 MIT"),
    "BetterEnd": ("partial", "LICENSE.ASSETS：gui/infusion.png 等 3 个文件为 CC BY-NC-SA 4.0，不涉及物品贴图；其余 MIT"),
    "Inspirations": ("partial", "TEXTURE_LICENSE：两张书架方块贴图为 CC BY-NC-SA 4.0，不涉及物品贴图；其余 MIT"),
    "Ex Nihilo: Creatio": ("partial", "README：Crushed Sky Stone 贴图衍生自 AE2，为 CC BY-NC-SA 3.0，需剔除；其余 MIT"),
    # 素材不可用
    "Roots 4": ("exclude", "ASSET_LICENSE：素材保留所有权利；README：MIT 不适用于素材"),
    "Mystical World": ("exclude", "ART_LICENSE：美术素材归各作者，保留所有权利"),
    "Defiled Lands": ("exclude", "LICENSE：MIT 仅限代码，素材未经作者许可不得复用"),
    "Create Big Cannons": ("exclude", "LICENSE：手绘贴图和模型为 CC BY-NC-SA 4.0（非商用）"),
    "Croptopia": ("exclude", "仓库 LICENSE.txt 现为 All rights reserved（2025），与 CurseForge 的 MIT 声明矛盾"),
    "MythicBotany": ("exclude", "LICENSE：textures 目录整体适用 Botania 许可，非 Apache"),
    # 附属模组，本体素材许可更严
    "AIOT Botania": ("caution", "仓库为 MIT 且无素材条款，但贴图很可能衍生自 Botania（自有许可）"),
    "AE2 Things [Forge]": ("caution", "仓库为 MIT 且无素材条款，但贴图可能衍生自 AE2（素材 CC BY-NC-SA 3.0）"),
    "Create: Diesel Generators": ("caution", "仓库为 MIT 且无素材条款，但可能沿用 Create 的素材（保留所有权利）"),
    "Theurgy": ("caution", "README：素材为 CC-BY-SA-4.0，带相同方式共享条款；逐文件许可见 REUSE.toml"),
    "Eidolon": ("caution", "仓库无许可文件；mods.toml 写 All rights reserved（可能是模板默认值），CurseForge 写 BSD"),
    # 仓库里没有许可全文，只有元数据声明
    "Extreme Reactors": ("ok_weak", "仓库无许可文件；mods.toml 声明 MIT"),
    "Nether Depths Upgrade": ("ok_weak", "仓库无许可文件；mods.toml 声明 MIT"),
    "Ocean's Delight": ("ok_weak", "仓库里的 LICENSE.txt 是 Forge 模板自带的；mods.toml 声明 MIT"),
    "Crabber's Delight": ("ok_weak", "仓库无许可文件；gradle.properties 声明 MIT"),
    "Expanded Delight": ("ok_weak", "仓库只有 NeoForge 模板的许可；gradle.properties 声明 MIT"),
    # 只有 CurseForge 页面的声明
    "Rustic": ("unverified", "仓库无许可文件，README 未提及许可"),
    "Better Survival mod": ("unverified", "仓库无许可文件，README 未提及许可"),
    "Farmer's Knives": ("unverified", "仓库无许可文件；README 里的 CC0 指的是模板"),
    "Dank Storage": ("unverified", "仓库无许可文件，mods.toml 的 license 字段写的是 yes"),
    "Food Expansion": ("unverified", "仓库 Lellson/Food-Expansion 无许可文件"),
}
REPO_FIX = {
    "Magic Bees": ("ForestryMC/MagicBees", "LICENSE 为 WTFPL，未见素材单独条款"),
    "Food Expansion": ("Lellson/Food-Expansion", None),
    "Iron Furnaces": ("Qelifern/IronFurnacesNeoForge", "NeoForge 仓库为 Apache 2.0；旧仓库 Qelifern/IronFurnaces 是 GPL-3.0，只能用新版"),
}


def main():
    raw = {e["name"]: e for e in json.loads(RAW.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    out = []
    for r in rows:
        e = raw[r["name"]]
        repo = e["repo"]
        verdict, note = VERDICTS.get(r["name"], ("ok_default", ""))
        if r["name"] in REPO_FIX:
            repo, fix_note = REPO_FIX[r["name"]]
            note = note or fix_note
        if verdict == "ok_default" and not note:
            lic = sorted({k for d, v in e["docs"].items() if "readme" not in d.lower() for k in v["kinds"]})
            note = f"仓库许可文件：{'/'.join(lic) or '?'}，未见素材单独条款"
        out.append({
            "project_id": r["project_id"], "slug": r["slug"], "name": r["name"], "downloads": r["downloads"],
            "cf_license": r["license"], "repo": f"https://github.com/{repo}", "verdict": verdict, "note": note,
        })
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    c = Counter(o["verdict"] for o in out)
    print(dict(c), "total", sum(c.values()))
    odd = [o["name"] for o in out if o["verdict"] == "ok_default" and "?" in o["note"]]
    print("ok_default without detected license:", odd)
    missing = [k for k in list(VERDICTS) + list(REPO_FIX) if k not in raw]
    print("names not in list:", missing)


if __name__ == "__main__":
    sys.exit(main())
