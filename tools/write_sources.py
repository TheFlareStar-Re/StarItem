"""生成随模型发布的训练素材来源与许可清单。

Reads mods_final.csv and dataset/clean/metadata.jsonl, writes DATA_SOURCES.md and docs/data_sources.csv.

Usage:
  python -X utf8 tools/write_sources.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSE_NAMES = {
    "MIT License": "MIT", "Apache License version 2.0": "Apache-2.0", "Creative Commons Attribution 3.0 Unported": "CC BY 3.0",
    "CC-BY-4.0": "CC BY 4.0", "CC-BY-NC-SA-4.0": "CC BY-NC-SA 4.0", "CC-BY-NC-SA-3.0": "CC BY-NC-SA 3.0", "CC0-1.0": "CC0 1.0",
    "-": "",
}
CATEGORY = {
    "ok_default": "宽松许可", "ok_explicit": "许可明确覆盖素材", "ok_nc": "仅限非商用",
    "ok_weak": "宽松许可（证据较弱）", "partial": "宽松许可（个别文件另有条款）",
}
# 模型实际用到的合成数据，数字来自各数据集的 metadata.jsonl
SYNTHETIC = [("synth2", "本项目的 LoRA 直接生成，每个物品 2～3 张"), ("many", "本项目的 LoRA 直接生成，每个物品约 20 张，筛选后保留"),
             ("manual", "外部图像生成模型出底稿，再由本项目的 LoRA 重画")]


def declared(text):
    text = text.removeprefix("Modrinth: ").strip()
    return LICENSE_NAMES.get(text, text)


def cell(text):
    return str(text).replace("|", "/").replace("\n", " ")


def main():
    mods = list(csv.DictReader((ROOT / "mods_final.csv").open(encoding="utf-8-sig")))
    meta = [json.loads(line) for line in (ROOT / "dataset" / "clean" / "metadata.jsonl").open(encoding="utf-8")]
    used = Counter(m["slug"] for m in meta if m["size"] == 16)
    rows = []
    for m in mods:
        if not used[m["slug"]]:
            continue  # 清单里有、但没有 16x16 物品贴图进入训练集的模组不列
        rows.append({
            "mod": m["name"], "author": m["author"], "sprites": used[m["slug"]], "declared_license": declared(m["cf_license"]),
            "category": CATEGORY[m["verdict"]], "review_note": m["note"],
            # 这一栏是模组的发布页；只在 Modrinth 上的模组给 Modrinth 的地址
            "curseforge": (f"https://modrinth.com/mod/{m['slug']}" if m["batch"].endswith("_mr")
                           else f"https://www.curseforge.com/minecraft/mc-mods/{m['slug']}"), "source": m["repo"],
            "verdict": m["verdict"], "project_id": m["project_id"], "slug": m["slug"],
        })
    rows.sort(key=lambda r: -r["sprites"])
    total = sum(r["sprites"] for r in rows)

    (ROOT / "docs").mkdir(exist_ok=True)
    with (ROOT / "docs" / "data_sources.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = Counter()
    for r in rows:
        by[r["category"]] += r["sprites"]
    out = ["# 训练素材来源与许可", "",
           "本文件列出 SIC-1.2 模型训练时用到的全部素材来源。机器可读的版本在 [docs/data_sources.csv](docs/data_sources.csv)。", "",
           "## 概览", "",
           f"- 手绘贴图：{total:,} 张 16×16 物品贴图，来自 {len(rows)} 个开源 Minecraft 模组。每个模组的素材许可都逐个仓库复核过，"
           "不是只看 CurseForge 页面上的声明（那里的许可通常只针对代码）。",
           "- 合成贴图：由图像生成模型按物品名称生成，不取自任何模组，详见文末。",
           "- 没有直接从原版 Minecraft 提取贴图，手绘素材全部取自下列模组的发布包。", "",
           "| 类别 | 贴图数 | 模组数 |", "|---|---|---|"]
    for name in dict.fromkeys(CATEGORY.values()):
        out.append(f"| {name} | {by[name]:,} | {sum(1 for r in rows if r['category'] == name)} |")
    out += ["", "## 收录规则", "",
            "- 收录：素材许可为 MIT、Apache-2.0、CC0、CC BY 等宽松许可的模组；以及 CC BY-NC、CC BY-NC-SA 等仅限非商用的许可"
            "（因此模型权重以 CC BY-NC-SA 4.0 发布，只能用于非商业用途，见 LICENSE-MODEL.md）。",
            "- 不收录：保留所有权利（ARR）、GPL/LGPL、禁止演绎（ND）的素材；仓库里找不到许可信息的模组；"
            "依附于素材许可更严格的模组的附属模组。",
            "- 许可是按模组仓库在复核时的状态判断的。如果你是某个模组的作者，认为这里的判断有误或希望移除你的素材，"
            "请在本项目仓库提 issue，我们会在下一个版本的训练集中去掉。", ""]

    def table(title, intro, picked):
        out.extend([f"## {title}", "", intro, "", "| 模组 | 作者 | 贴图数 | 复核说明 | 源码 |", "|---|---|---|---|---|"])
        for r in picked:
            out.append(f"| [{cell(r['mod'])}]({r['curseforge']}) | {cell(r['author'])} | {r['sprites']} | {cell(r['review_note'])} | [仓库]({r['source']}) |")
        out.append("")

    table("仅限非商用的素材", "下列模组的素材采用非商用许可，按各自许可的要求在此署名。",
          [r for r in rows if r["verdict"] == "ok_nc"])
    table("许可明确覆盖素材的模组", "下列模组在仓库里明确写了素材的许可，其中 CC BY 的素材按要求在此署名。",
          [r for r in rows if r["verdict"] == "ok_explicit"])
    table("个别文件另有条款的模组", "下列模组的大部分素材是宽松许可，少数文件另有条款，处理方式见复核说明。",
          [r for r in rows if r["verdict"] == "partial"])
    table("许可证据较弱的模组", "下列模组的仓库里没有独立的许可文件，许可依据是构建配置里的声明。",
          [r for r in rows if r["verdict"] == "ok_weak"])

    out += ["## 全部模组", "", "按进入训练集的贴图数排序。", "",
            "| 模组 | 作者 | 贴图数 | 声明的许可 | 类别 | 源码 |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| [{cell(r['mod'])}]({r['curseforge']}) | {cell(r['author'])} | {r['sprites']} | {cell(r['declared_license']) or '见复核说明'} | "
                   f"{r['category']} | [仓库]({r['source']}) |")

    kept = [json.loads(line) for line in (ROOT / "dataset" / "manual" / "metadata.jsonl").open(encoding="utf-8")]
    drafts = sum(1 for m in kept if m["kind"] == "manual")
    drafted = len({m["name_en"] for m in kept})
    out += ["", "## 合成贴图", "",
            "合成贴图不取自任何模组，是按 4,046 个日常物品的名称清单（`synth/concepts/`）生成的。", "",
            "| 数据集 | 贴图数 | 物品数 | 生成方式 |", "|---|---|---|---|"]
    for name, how in SYNTHETIC:
        items = [json.loads(line) for line in (ROOT / "dataset" / name / "metadata.jsonl").open(encoding="utf-8")]
        out.append(f"| {name} | {len(items):,} | {len({i['name_en'] for i in items}):,} | {how} |")
    out += ["",
            "- “本项目的 LoRA”是在 Stable Diffusion XL 上、只用上面列出的手绘贴图训练的风格 LoRA，它让 SDXL 直接画出 16×16 网格的贴图。",
            f"- 外部图像生成模型只用于 {drafted:,} 个 LoRA 画不好的物品，每个物品 3 张底稿，共 {drafts:,} 张；底稿缩成 16×16 后进入训练集，"
            "并由 LoRA 以它为底各重画成多张变体。",
            "- 贴图的筛选用了 CLIP（判断生成的图是否像它的名字）。", "",
            "## 用到的模型和组件", "",
            "| 名称 | 用途 | 许可 |", "|---|---|---|",
            "| [Stable Diffusion XL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 生成合成贴图（仅训练阶段） | CreativeML Open RAIL++-M |",
            "| [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | 同上 | MIT |",
            "| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | 筛选合成贴图（仅训练阶段） | MIT |",
            "| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 文本编码器，随模型一起分发 | MIT |",
            "| [U²-Net-p](https://github.com/xuebinqin/U-2-Net) | 从参考照片里抠出物体，随客户端和模型一起分发（ONNX 文件取自 [rembg](https://github.com/danielgatis/rembg) 的发布页，原样分发） | [Apache-2.0](licenses/Apache-2.0.txt) |",
            "| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 客户端推理 | MIT |", ""]
    (ROOT / "DATA_SOURCES.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"{len(rows)} mods, {total} sprites -> DATA_SOURCES.md, docs/data_sources.csv")
    print(dict(by))


if __name__ == "__main__":
    main()
