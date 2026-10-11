# 训练素材来源与许可

本文件列出 SIC-1.2 模型训练时用到的全部素材来源。机器可读的版本在 [docs/data_sources.csv](docs/data_sources.csv)。

## 概览

- 手绘贴图：17,567 张 16×16 物品贴图，来自 200 个开源 Minecraft 模组。每个模组的素材许可都逐个仓库复核过，不是只看 CurseForge 页面上的声明（那里的许可通常只针对代码）。
- 合成贴图：由图像生成模型按物品名称生成，不取自任何模组，详见文末。
- 没有直接从原版 Minecraft 提取贴图，手绘素材全部取自下列模组的发布包。

| 类别 | 贴图数 | 模组数 |
|---|---|---|
| 宽松许可 | 12,076 | 152 |
| 许可明确覆盖素材 | 1,932 | 16 |
| 仅限非商用 | 2,267 | 14 |
| 宽松许可（证据较弱） | 753 | 14 |
| 宽松许可（个别文件另有条款） | 539 | 4 |

## 收录规则

- 收录：素材许可为 MIT、Apache-2.0、CC0、CC BY 等宽松许可的模组；以及 CC BY-NC、CC BY-NC-SA 等仅限非商用的许可（因此模型权重以 CC BY-NC-SA 4.0 发布，只能用于非商业用途，见 LICENSE-MODEL.md）。
- 不收录：保留所有权利（ARR）、GPL/LGPL、禁止演绎（ND）的素材；仓库里找不到许可信息的模组；依附于素材许可更严格的模组的附属模组。
- 许可是按模组仓库在复核时的状态判断的。如果你是某个模组的作者，认为这里的判断有误或希望移除你的素材，请在本项目仓库提 issue，我们会在下一个版本的训练集中去掉。

## 仅限非商用的素材

下列模组的素材采用非商用许可，按各自许可的要求在此署名。

| 模组 | 作者 | 贴图数 | 复核说明 | 源码 |
|---|---|---|---|---|
| [ExtraDelight](https://www.curseforge.com/minecraft/mc-mods/extradelight) | Lance5057 | 817 | 仓库无许可文件；gradle.properties、mods.toml 和 Modrinth 页面均声明 CC BY-NC-SA 4.0（非商用）；农夫乐事的附属 | [仓库](https://github.com/Lance5057/ExtraDelight) |
| [Kaleidoscope Grilling](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-grilling) | breezeth | 224 | LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0；森罗物语厨房的附属 | [仓库](https://github.com/breezeth-CN/KaleidoscopeGrilling) |
| [Kaleidoscope Cookery](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-cookery) | YSBB | 173 | LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0 | [仓库](https://github.com/KaleidoscopeMods/KaleidoscopeCookery) |
| [Butchercraft](https://www.curseforge.com/minecraft/mc-mods/butchercraft) | Lance5057 | 163 | LICENSE.md：整个项目 CC BY-NC-SA 4.0 | [仓库](https://github.com/Lance5057/Butchercraft) |
| [Quark](https://www.curseforge.com/minecraft/mc-mods/quark) | Vazkii | 145 | LICENSE：整个模组 CC BY-NC-SA 3.0 | [仓库](https://github.com/VazkiiMods/Quark) |
| [Oritech](https://www.curseforge.com/minecraft/mc-mods/oritech) | rearth | 129 | LICENSE 为 CC0；README：部分素材来自 techarium（CC BY-NC 4.0）和 malcolmriley（CC BY 4.0），需署名 | [仓库](https://github.com/Rearth/Oritech) |
| [Bakeries](https://www.curseforge.com/minecraft/mc-mods/bakeries) | Renyigesai | 129 | LICENSE-ASSETS.txt：资源文件 CC BY-NC-SA 4.0 | [仓库](https://github.com/Renyigesai/bakery) |
| [Applied Energistics 2](https://www.curseforge.com/minecraft/mc-mods/applied-energistics-2) | thetechnici4n | 127 | README：贴图和模型 CC BY-NC-SA 3.0（代码 LGPL） | [仓库](https://github.com/AppliedEnergistics/Applied-Energistics-2) |
| [Armourer's Workshop](https://www.curseforge.com/minecraft/mc-mods/armourers-workshop) | RiskyKen | 87 | LICENSE：CC BY-NC-SA 3.0 | [仓库](https://github.com/Armourers-Workshop/Armourers-Workshop) |
| [Kaleidoscope Tavern](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-tavern) | TartaricAcid | 70 | LICENSE-ASSETS：资源文件 CC BY-NC-SA 4.0 | [仓库](https://github.com/KaleidoscopeMods/KaleidoscopeTavern) |
| [Kaleidoscope：Dimensions wine](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-dimensions-wine) | wither_redstone | 70 | LICENSE.md：assets 目录 CC BY-NC-SA 4.0（森罗物语酒馆的附属） | [仓库](https://github.com/WitherRedstone/Kaleidoscope-DimensionsWine) |
| [Biomancy](https://www.curseforge.com/minecraft/mc-mods/biomancy) | Elenterius | 62 | LICENSE：美术素材 CC BY-NC 4.0（代码 MIT） | [仓库](https://github.com/Elenterius/Biomancy) |
| [Create Big Cannons](https://www.curseforge.com/minecraft/mc-mods/create-big-cannons) | rbasamoyai | 45 | LICENSE：手绘贴图和模型 CC BY-NC-SA 4.0 | [仓库](https://github.com/Cannoneers-of-Create/CreateBigCannons) |
| [Legume Delight](https://www.curseforge.com/minecraft/mc-mods/legume-delight) | pawjwp | 26 | LICENSE.txt：整个项目 CC BY-NC-SA 4.0；农夫乐事的附属 | [仓库](https://github.com/pawjwp/Legume-Delight) |

## 许可明确覆盖素材的模组

下列模组在仓库里明确写了素材的许可，其中 CC BY 的素材按要求在此署名。

| 模组 | 作者 | 贴图数 | 复核说明 | 源码 |
|---|---|---|---|---|
| [Modern Industrialization](https://www.curseforge.com/minecraft/mc-mods/modern-industrialization) | thetechnici4n | 553 | README：素材全部 CC0；部分贴图改自 GregTech 等他人作品 | [仓库](https://github.com/AztechMC/Modern-Industrialization) |
| [Blood Magic](https://www.curseforge.com/minecraft/mc-mods/blood-magic) | WayofTime | 326 | LICENSE：整个项目 CC BY 4.0，需署名 | [仓库](https://github.com/WayofTime/BloodMagic) |
| [Occultism](https://www.curseforge.com/minecraft/mc-mods/occultism) | kli_kli | 254 | THIRD_PARTY_NOTICES：未列出的素材随项目 MIT；列出的第三方图标为 CC BY 3.0，需署名 | [仓库](https://github.com/klikli-dev/occultism) |
| [Tinkers Construct](https://www.curseforge.com/minecraft/mc-mods/tinkers-construct) | mDiyo | 253 | README：代码、贴图、二进制均为 MIT | [仓库](https://github.com/SlimeKnights/TinkersConstruct) |
| [OpenComputers](https://www.curseforge.com/minecraft/mc-mods/opencomputers) | Sangar | 136 | LICENSE：贴图为 CC0，除非另有标注 | [仓库](https://github.com/MightyPirates/OpenComputers) |
| [Natura](https://www.curseforge.com/minecraft/mc-mods/natura) | mDiyo | 96 | LICENSE.md：贴图 CC BY 3.0，需署名 | [仓库](https://github.com/progwml6/Natura) |
| [Light Meals](https://www.curseforge.com/minecraft/mc-mods/light-meals) | Hashtag233 | 78 | 仓库根目录和 resources 目录下的 LICENSE 均为 MIT | [仓库](https://github.com/Hashtag233/LightMeals) |
| [EvilCraft](https://www.curseforge.com/minecraft/mc-mods/evilcraft) | kroeser | 77 | README：代码和图片 CC BY 4.0，需署名（CurseForge 写的是 Public Domain） | [仓库](https://github.com/CyclopsMC/EvilCraft) |
| [Paper Kite Manor](https://www.curseforge.com/minecraft/mc-mods/paper-kite-manor) | KaziCat | 62 | LICENSE-ASSETS：资源文件 MIT | [仓库](https://github.com/MasterMarisa/PaperKiteManor) |
| [Angling](https://modrinth.com/mod/angling) | EightSidedSquare | 23 | LICENSE：CC0 | [仓库](https://github.com/8s2/Angling) |
| [Integrated Dynamics](https://www.curseforge.com/minecraft/mc-mods/integrated-dynamics) | kroeser | 21 | README：代码和图片均为 MIT | [仓库](https://github.com/CyclopsMC/IntegratedDynamics) |
| [Vegehenna: A vegetarian's heaven?](https://modrinth.com/mod/vegehenna) | ivangeevo | 16 | LICENSE：整个项目 CC BY 4.0，需署名 | [仓库](https://github.com/ivangeevo/vegehenna) |
| [Scannable](https://www.curseforge.com/minecraft/mc-mods/scannable) | Sangar | 11 | LICENSE：贴图为 CC0，除非另有标注 | [仓库](https://github.com/MightyPirates/Scannable) |
| [Farmer's Crops](https://modrinth.com/mod/farmers-crops) | JCS | 11 | LICENSE：CC0 | [仓库](https://github.com/JCS-Mecabricks/Farmers-Crops) |
| [Tinkers' Levelling Addon](https://www.curseforge.com/minecraft/mc-mods/tinkers-levelling-addon) | Pyre540 | 8 | README：代码、贴图、二进制均为 MIT | [仓库](https://github.com/Pyre540/tinkers-levelling-addon) |
| [Tinkers' Mechworks](https://www.curseforge.com/minecraft/mc-mods/tinkers-mechworks) | mDiyo | 7 | README：贴图 CC BY 3.0，需署名 | [仓库](https://github.com/SlimeKnights/TinkersMechworks) |

## 个别文件另有条款的模组

下列模组的大部分素材是宽松许可，少数文件另有条款，处理方式见复核说明。

| 模组 | 作者 | 贴图数 | 复核说明 | 源码 |
|---|---|---|---|---|
| [BetterEnd](https://www.curseforge.com/minecraft/mc-mods/betterend) | Quiqueck | 175 | LICENSE.ASSETS：gui/infusion.png 等 3 个文件为 CC BY-NC-SA 4.0，不涉及物品贴图；其余 MIT | [仓库](https://github.com/quiqueck/BetterEnd) |
| [Inspirations](https://www.curseforge.com/minecraft/mc-mods/inspirations) | KnightMiner | 152 | TEXTURE_LICENSE：两张书架方块贴图为 CC BY-NC-SA 4.0，不涉及物品贴图；其余 MIT | [仓库](https://github.com/KnightMiner/Inspirations) |
| [BetterNether](https://www.curseforge.com/minecraft/mc-mods/betternether) | Quiqueck | 147 | LICENSE.ASSETS：item/*_boat.png、*_chest_boat.png 等为 CC BY-NC-SA 4.0（不商用，保留）；其余 MIT | [仓库](https://github.com/quiqueck/BetterNether) |
| [Ex Nihilo: Creatio](https://www.curseforge.com/minecraft/mc-mods/ex-nihilo-creatio) | BloodWorkXGaming | 65 | README：Crushed Sky Stone 贴图衍生自 AE2，为 CC BY-NC-SA 3.0，需剔除；其余 MIT | [仓库](https://github.com/BloodyMods/ExNihiloCreatio) |

## 许可证据较弱的模组

下列模组的仓库里没有独立的许可文件，许可依据是构建配置里的声明。

| 模组 | 作者 | 贴图数 | 复核说明 | 源码 |
|---|---|---|---|---|
| [Hearth and Harvest](https://www.curseforge.com/minecraft/mc-mods/hearth-and-harvest) | AlabasterLeking | 104 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/AlabasterLeking/Hearth-And-Harvest) |
| [Crabber's Delight](https://www.curseforge.com/minecraft/mc-mods/crabbers-delight) | AlabasterLeking | 84 | 仓库无许可文件；gradle.properties 声明 MIT | [仓库](https://github.com/AlabasterLeking/Crabbers-Delight) |
| [Critters Delight](https://www.curseforge.com/minecraft/mc-mods/critters-delight) | DakotaPrideModding | 83 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/dakotapride/critters-delight) |
| [Yeast 'n Feast](https://www.curseforge.com/minecraft/mc-mods/yeast-n-feast) | AstralyaStudios | 81 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT | [仓库](https://github.com/AstralyaStudios/YeastNFeast) |
| [Farmers Expanse](https://modrinth.com/mod/farmers-expanse) | patrickrich07 | 79 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/Greddode/Farmers-Expanse) |
| [Expanded Delight](https://www.curseforge.com/minecraft/mc-mods/expanded-delight) | ianm1647 | 72 | 仓库只有 NeoForge 模板的许可；gradle.properties 声明 MIT | [仓库](https://github.com/ianm1647/expandeddelight) |
| [Nether Depths Upgrade](https://www.curseforge.com/minecraft/mc-mods/nether-depths-upgrade) | scouter567 | 68 | 仓库无许可文件；mods.toml 声明 MIT | [仓库](https://github.com/Scouter456/Nether_Depths_Upgrade) |
| [Peruvian's Delight](https://www.curseforge.com/minecraft/mc-mods/peruvians-delight) | GiulianoMMPP | 57 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/GiulianoMaPe/PeruviansDelight-1.20.1-Forge) |
| [Veggie Delight](https://modrinth.com/mod/veggie-delight) | ElectroArtifex | 28 | 仓库无许可文件；fabric.mod.json 声明 CC BY 4.0，Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/ElectroArtifex/veggiedelight-1.20.1) |
| [Hot Kettles](https://www.curseforge.com/minecraft/mc-mods/hotkettles) | slainlight | 26 | 仓库无许可文件；fabric.mod.json、mods.toml 和 Modrinth 页面声明 MIT | [仓库](https://github.com/matthewperiut/hotkettles) |
| [Ocean's Delight](https://www.curseforge.com/minecraft/mc-mods/oceans-delight) | scouter567 | 25 | 仓库里的 LICENSE.txt 是 Forge 模板自带的；mods.toml 声明 MIT | [仓库](https://github.com/Scouter456/Oceans_Delight) |
| [Large Meals](https://modrinth.com/mod/large-meals) | KreloX | 18 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/KreloX/Large-Meals) |
| [Spanish Delight](https://modrinth.com/mod/spanish-delight) | DevJulen | 17 | 仓库无许可文件；gradle.properties 和 Modrinth 页面声明 MIT；农夫乐事的附属 | [仓库](https://github.com/DevJulen/SpanishDelight-Forge) |
| [Extreme Reactors](https://www.curseforge.com/minecraft/mc-mods/extreme-reactors) | ZeroNoRyouki | 11 | 仓库无许可文件；mods.toml 声明 MIT | [仓库](https://github.com/ZeroNoRyouki/ExtremeReactors2) |

## 全部模组

按进入训练集的贴图数排序。

| 模组 | 作者 | 贴图数 | 声明的许可 | 类别 | 源码 |
|---|---|---|---|---|---|
| [ExtraDelight](https://www.curseforge.com/minecraft/mc-mods/extradelight) | Lance5057 | 817 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/Lance5057/ExtraDelight) |
| [Modern Industrialization](https://www.curseforge.com/minecraft/mc-mods/modern-industrialization) | thetechnici4n | 553 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/AztechMC/Modern-Industrialization) |
| [Photosynthesis](https://modrinth.com/mod/photosynthesis) | Zuiron | 548 | MIT | 宽松许可 | [仓库](https://github.com/Zuiron/Photosynthesis) |
| [Mystical Agriculture](https://www.curseforge.com/minecraft/mc-mods/mystical-agriculture) | BlakeBr0 | 540 | MIT | 宽松许可 | [仓库](https://github.com/BlakeBr0/MysticalAgriculture) |
| [Spartan Weaponry](https://www.curseforge.com/minecraft/mc-mods/spartan-weaponry) | ObliviousSpartan | 495 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/ObliviousSpartan/SpartanWeaponry) |
| [Tech Reborn](https://www.curseforge.com/minecraft/mc-mods/techreborn) | modmuss50 | 414 | MIT | 宽松许可 | [仓库](https://github.com/TechReborn/TechReborn) |
| [Eternal Starlight](https://www.curseforge.com/minecraft/mc-mods/eternal-starlight) | LeoLezury | 414 | MIT | 宽松许可 | [仓库](https://github.com/LeoMinecraftModding/eternal-starlight) |
| [Cosmopolitan](https://www.curseforge.com/minecraft/mc-mods/cosmopolitan) | Gumillea | 397 | MIT | 宽松许可 | [仓库](https://github.com/Gumillea/Cosmopolitan) |
| [Sakura Mod](https://modrinth.com/mod/sakura_mod) | Syameimaru_Zheng | 340 | MIT | 宽松许可 | [仓库](https://github.com/0999312/Sakura_mod) |
| [Blood Magic](https://www.curseforge.com/minecraft/mc-mods/blood-magic) | WayofTime | 326 | CC BY 4.0 | 许可明确覆盖素材 | [仓库](https://github.com/WayofTime/BloodMagic) |
| [Immortalers Delight](https://www.curseforge.com/minecraft/mc-mods/immortalers-delight) | Renyigesai | 299 | MIT | 宽松许可 | [仓库](https://github.com/Renyigesai/immortalers_delight) |
| [Occultism](https://www.curseforge.com/minecraft/mc-mods/occultism) | kli_kli | 254 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/klikli-dev/occultism) |
| [Tinkers Construct](https://www.curseforge.com/minecraft/mc-mods/tinkers-construct) | mDiyo | 253 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/SlimeKnights/TinkersConstruct) |
| [Silent Gear](https://www.curseforge.com/minecraft/mc-mods/silent-gear) | SilentChaos512 | 251 | MIT | 宽松许可 | [仓库](https://github.com/SilentChaos512/Silent-Gear) |
| [Cyclic](https://www.curseforge.com/minecraft/mc-mods/cyclic) | Lothrazar | 245 | 见复核说明 | 宽松许可 | [仓库](https://github.com/Lothrazar/Cyclic) |
| [Amethyst Imbuement](https://www.curseforge.com/minecraft/mc-mods/amethyst-imbuement) | fzzyhmstrs | 245 | MIT | 宽松许可 | [仓库](https://github.com/fzzyhmstrs/ai) |
| [Embers Rekindled](https://www.curseforge.com/minecraft/mc-mods/embers-rekindled) | BordListian | 233 | MIT | 宽松许可 | [仓库](https://github.com/RCXcrafter/EmbersRekindled) |
| [Bagel's Baking](https://modrinth.com/mod/bagels-baking) | hugeblank | 233 | MIT | 宽松许可 | [仓库](https://github.com/hugeblank/Bagels_Baking) |
| [Actually Additions](https://www.curseforge.com/minecraft/mc-mods/actually-additions) | Ellpeck | 227 | 见复核说明 | 宽松许可 | [仓库](https://github.com/Ellpeck/ActuallyAdditions) |
| [Kaleidoscope Grilling](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-grilling) | breezeth | 224 | BSD-3-Clause | 仅限非商用 | [仓库](https://github.com/breezeth-CN/KaleidoscopeGrilling) |
| [Hibernal Herbs](https://www.curseforge.com/minecraft/mc-mods/hibernal-herbs) | DakotaPrideModding | 222 | MIT | 宽松许可 | [仓库](https://github.com/DakotaPride/Hibernal-Herbs) |
| [Modern Delight](https://www.curseforge.com/minecraft/mc-mods/modern-delight) | zombie_cute | 216 | MIT | 宽松许可 | [仓库](https://github.com/zombiecute/ModernDelight) |
| [Collector's Reap](https://www.curseforge.com/minecraft/mc-mods/collectors-reap) | brnbrd | 200 | MIT | 宽松许可 | [仓库](https://github.com/brnbrd/CollectorsReap) |
| [Mekanism](https://www.curseforge.com/minecraft/mc-mods/mekanism) | bradyaidanc | 198 | MIT | 宽松许可 | [仓库](https://github.com/mekanism/Mekanism) |
| [Steve's Carts Reborn](https://www.curseforge.com/minecraft/mc-mods/steves-carts-reborn) | modmuss50 | 185 | MIT | 宽松许可 | [仓库](https://github.com/TechReborn/StevesCarts) |
| [BetterEnd](https://www.curseforge.com/minecraft/mc-mods/betterend) | Quiqueck | 175 | MIT | 宽松许可（个别文件另有条款） | [仓库](https://github.com/quiqueck/BetterEnd) |
| [Kaleidoscope Cookery](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-cookery) | YSBB | 173 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/KaleidoscopeMods/KaleidoscopeCookery) |
| [Butchercraft](https://www.curseforge.com/minecraft/mc-mods/butchercraft) | Lance5057 | 163 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/Lance5057/Butchercraft) |
| [Regions Unexplored](https://www.curseforge.com/minecraft/mc-mods/regions-unexplored) | Apollo | 159 | MIT | 宽松许可 | [仓库](https://github.com/Apollounknowndev/RegionsUnexplored) |
| [Just Dire Things](https://www.curseforge.com/minecraft/mc-mods/just-dire-things) | Direwolf20 | 159 | MIT | 宽松许可 | [仓库](https://github.com/Direwolf20-MC/JustDireThings) |
| [Delightful](https://www.curseforge.com/minecraft/mc-mods/delightful) | brnbrd | 156 | MIT | 宽松许可 | [仓库](https://github.com/brnbrd/Delightful) |
| [Inspirations](https://www.curseforge.com/minecraft/mc-mods/inspirations) | KnightMiner | 152 | MIT | 宽松许可（个别文件另有条款） | [仓库](https://github.com/KnightMiner/Inspirations) |
| [BetterNether](https://www.curseforge.com/minecraft/mc-mods/betternether) | Quiqueck | 147 | MIT | 宽松许可（个别文件另有条款） | [仓库](https://github.com/quiqueck/BetterNether) |
| [Quark](https://www.curseforge.com/minecraft/mc-mods/quark) | Vazkii | 145 | 见复核说明 | 仅限非商用 | [仓库](https://github.com/VazkiiMods/Quark) |
| [Farmer's Delight](https://www.curseforge.com/minecraft/mc-mods/farmers-delight) | vectorwing | 142 | MIT | 宽松许可 | [仓库](https://github.com/vectorwing/FarmersDelight) |
| [Ender IO](https://www.curseforge.com/minecraft/mc-mods/ender-io) | crazypants_mc_the_second | 141 | Public Domain | 宽松许可 | [仓库](https://github.com/Team-EnderIO/EnderIO) |
| [Kitchen Karrot](https://www.curseforge.com/minecraft/mc-mods/kitchen-karrot) | Zoshsgahdnkc | 137 | MIT | 宽松许可 | [仓库](https://github.com/Zoshsgahdnkc/kitchenkarrot) |
| [OpenComputers](https://www.curseforge.com/minecraft/mc-mods/opencomputers) | Sangar | 136 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/MightyPirates/OpenComputers) |
| [Oritech](https://www.curseforge.com/minecraft/mc-mods/oritech) | rearth | 129 | CC BY 4.0 | 仅限非商用 | [仓库](https://github.com/Rearth/Oritech) |
| [Brazilian Delight](https://www.curseforge.com/minecraft/mc-mods/braziliandelight) | DeltaboxLabs | 129 | MIT | 宽松许可 | [仓库](https://github.com/danbrown/braziliandelight) |
| [Bakeries](https://www.curseforge.com/minecraft/mc-mods/bakeries) | Renyigesai | 129 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/Renyigesai/bakery) |
| [Applied Energistics 2](https://www.curseforge.com/minecraft/mc-mods/applied-energistics-2) | thetechnici4n | 127 | 见复核说明 | 仅限非商用 | [仓库](https://github.com/AppliedEnergistics/Applied-Energistics-2) |
| [Rustic Delight](https://www.curseforge.com/minecraft/mc-mods/rustic-delight) | PhantomWing | 123 | MIT | 宽松许可 | [仓库](https://github.com/PhantomWing/RusticDelight) |
| [Enhanced Farming](https://www.curseforge.com/minecraft/mc-mods/enhanced-farming) | Mrbysco | 122 | MIT | 宽松许可 | [仓库](https://github.com/Mrbysco/Enhanced-Farming) |
| [Foodables](https://modrinth.com/mod/foodables) | TruthSt | 121 | MIT | 宽松许可 | [仓库](https://github.com/CanINotUseTruth/Foodables) |
| [Industrial Foregoing](https://www.curseforge.com/minecraft/mc-mods/industrial-foregoing) | Buuz135 | 119 | MIT | 宽松许可 | [仓库](https://github.com/InnovativeOnlineIndustries/Industrial-Foregoing) |
| [UsefulFood Reborn](https://www.curseforge.com/minecraft/mc-mods/usefulfood-reborn) | skniro | 119 | MIT | 宽松许可 | [仓库](https://github.com/skniro/UsefulFood-Reborn) |
| [Ube's Delight](https://www.curseforge.com/minecraft/mc-mods/ubes-delight) | ChefMooon | 117 | MIT | 宽松许可 | [仓库](https://github.com/ChefMooon/ubes-delight) |
| [Marium's Soulslike Weaponry](https://www.curseforge.com/minecraft/mc-mods/mariums-soulslike-weaponry) | mariumbacchus | 111 | CC0 1.0 | 宽松许可 | [仓库](https://github.com/mariumbacchus/Soulslike-Weaponry) |
| [ProjectE](https://www.curseforge.com/minecraft/mc-mods/projecte) | SinKillerJ | 107 | MIT | 宽松许可 | [仓库](https://github.com/sinkillerj/ProjectE) |
| [Hearth and Harvest](https://www.curseforge.com/minecraft/mc-mods/hearth-and-harvest) | AlabasterLeking | 104 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/AlabasterLeking/Hearth-And-Harvest) |
| [Raven Coffee Mod](https://www.curseforge.com/minecraft/mc-mods/raven-coffee) | TheWandererRaven | 100 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/TheWandererRaven/RavenCoffeeMod) |
| [Natura](https://www.curseforge.com/minecraft/mc-mods/natura) | mDiyo | 96 | CC BY 3.0 | 许可明确覆盖素材 | [仓库](https://github.com/progwml6/Natura) |
| [Artifacts](https://www.curseforge.com/minecraft/mc-mods/artifacts) | ochotonida | 95 | MIT | 宽松许可 | [仓库](https://github.com/ochotonida/artifacts) |
| [NuclearCraft](https://www.curseforge.com/minecraft/mc-mods/nuclearcraft-mod) | tomdodd4598 | 94 | MIT | 宽松许可 | [仓库](https://github.com/tomdodd4598/NuclearCraft) |
| [Security Craft](https://www.curseforge.com/minecraft/mc-mods/security-craft) | Geforce132 | 93 | MIT | 宽松许可 | [仓库](https://github.com/Geforce132/SecurityCraft) |
| [Trail&Tales Delight](https://www.curseforge.com/minecraft/mc-mods/trail-tales-delight) | shuixingshow | 89 | MIT | 宽松许可 | [仓库](https://github.com/Vzling/Trail-And-Tales-Delight) |
| [Armourer's Workshop](https://www.curseforge.com/minecraft/mc-mods/armourers-workshop) | RiskyKen | 87 | CC BY-NC-SA 3.0 | 仅限非商用 | [仓库](https://github.com/Armourers-Workshop/Armourers-Workshop) |
| [Veggies Delight (A Farmer's Delight Add-on)](https://www.curseforge.com/minecraft/mc-mods/veggies-delight) | Eggmain | 87 | MIT | 宽松许可 | [仓库](https://github.com/MehdiNoui/VeggiesDelight) |
| [RFTools](https://www.curseforge.com/minecraft/mc-mods/rftools) | McJty | 85 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFTools) |
| [Magic Bees](https://www.curseforge.com/minecraft/mc-mods/magic-bees) | MysteriousAges | 85 | WTFPL | 宽松许可 | [仓库](https://github.com/ForestryMC/MagicBees) |
| [Fright's Delight](https://www.curseforge.com/minecraft/mc-mods/frights-delight) | ChefMooon | 85 | MIT | 宽松许可 | [仓库](https://github.com/ChefMooon/frights-delight) |
| [Crabber's Delight](https://www.curseforge.com/minecraft/mc-mods/crabbers-delight) | AlabasterLeking | 84 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/AlabasterLeking/Crabbers-Delight) |
| [Critters Delight](https://www.curseforge.com/minecraft/mc-mods/critters-delight) | DakotaPrideModding | 83 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/dakotapride/critters-delight) |
| [Tasty Supplies](https://modrinth.com/mod/tasty-supplies) | OneShrimpADay | 82 | MIT | 宽松许可 | [仓库](https://github.com/atomic-junky/tasty-supplies) |
| [Yeast 'n Feast](https://www.curseforge.com/minecraft/mc-mods/yeast-n-feast) | AstralyaStudios | 81 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/AstralyaStudios/YeastNFeast) |
| [Nature's Aura](https://www.curseforge.com/minecraft/mc-mods/natures-aura) | Ellpeck | 79 | MIT | 宽松许可 | [仓库](https://github.com/Ellpeck/NaturesAura) |
| [Farmers Expanse](https://modrinth.com/mod/farmers-expanse) | patrickrich07 | 79 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/Greddode/Farmers-Expanse) |
| [Light Meals](https://www.curseforge.com/minecraft/mc-mods/light-meals) | Hashtag233 | 78 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/Hashtag233/LightMeals) |
| [EvilCraft](https://www.curseforge.com/minecraft/mc-mods/evilcraft) | kroeser | 77 | Public Domain | 许可明确覆盖素材 | [仓库](https://github.com/CyclopsMC/EvilCraft) |
| [Kawaii Dishes](https://www.curseforge.com/minecraft/mc-mods/kawaii-dishes) | roseisproot | 76 | MIT | 宽松许可 | [仓库](https://github.com/OHakimen/KawaiiDishes) |
| [Exquisito](https://www.curseforge.com/minecraft/mc-mods/exquisito) | Gumillea | 73 | MIT | 宽松许可 | [仓库](https://github.com/Gumillea/Exquisito) |
| [Storage Drawers](https://www.curseforge.com/minecraft/mc-mods/storage-drawers) | Texelsaur | 72 | MIT | 宽松许可 | [仓库](https://github.com/jaquadro/StorageDrawers) |
| [Expanded Delight](https://www.curseforge.com/minecraft/mc-mods/expanded-delight) | ianm1647 | 72 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/ianm1647/expandeddelight) |
| [Crafty Cuisine - Vanilla Inspired Food](https://modrinth.com/mod/crafty-cuisine) | randommcsomethin | 72 | MIT | 宽松许可 | [仓库](https://github.com/RandomMcSomethin/Crafty-Cuisine) |
| [Kaleidoscope Tavern](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-tavern) | TartaricAcid | 70 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/KaleidoscopeMods/KaleidoscopeTavern) |
| [Kaleidoscope：Dimensions wine](https://www.curseforge.com/minecraft/mc-mods/kaleidoscope-dimensions-wine) | wither_redstone | 70 | CC BY 4.0 | 仅限非商用 | [仓库](https://github.com/WitherRedstone/Kaleidoscope-DimensionsWine) |
| [Nether Depths Upgrade](https://www.curseforge.com/minecraft/mc-mods/nether-depths-upgrade) | scouter567 | 68 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/Scouter456/Nether_Depths_Upgrade) |
| [Sushi Go Crafting](https://www.curseforge.com/minecraft/mc-mods/sushigocrafting) | Buuz135 | 67 | MIT | 宽松许可 | [仓库](https://github.com/Buuz135/SushiGoCrafting) |
| [Mekanism Tools](https://www.curseforge.com/minecraft/mc-mods/mekanism-tools) | bradyaidanc | 66 | MIT | 宽松许可 | [仓库](https://github.com/mekanism/Mekanism) |
| [Ex Nihilo: Creatio](https://www.curseforge.com/minecraft/mc-mods/ex-nihilo-creatio) | BloodWorkXGaming | 65 | MIT | 宽松许可（个别文件另有条款） | [仓库](https://github.com/BloodyMods/ExNihiloCreatio) |
| [Gone Fishing!](https://modrinth.com/mod/gone-fishing) | Patbox | 65 | MIT | 宽松许可 | [仓库](https://github.com/patbox/gone-fishing) |
| [RFTools Dimensions](https://www.curseforge.com/minecraft/mc-mods/rftools-dimensions) | McJty | 62 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsDimensions) |
| [Biomancy](https://www.curseforge.com/minecraft/mc-mods/biomancy) | Elenterius | 62 | MIT | 仅限非商用 | [仓库](https://github.com/Elenterius/Biomancy) |
| [Fishing 101](https://modrinth.com/mod/fishing-101) | Macck209 | 62 | MIT | 宽松许可 | [仓库](https://github.com/Macck209/Fishing101) |
| [Paper Kite Manor](https://www.curseforge.com/minecraft/mc-mods/paper-kite-manor) | KaziCat | 62 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/MasterMarisa/PaperKiteManor) |
| [End's Delight](https://www.curseforge.com/minecraft/mc-mods/ends-delight) | foggyhillside | 61 | MIT | 宽松许可 | [仓库](https://github.com/FoggyHillside/End-s-Delight) |
| [Project Red - Exploration](https://www.curseforge.com/minecraft/mc-mods/project-red-exploration) | MrTJP | 60 | MIT | 宽松许可 | [仓库](https://github.com/MrTJP/ProjectRed) |
| [Pickle Tweaks](https://www.curseforge.com/minecraft/mc-mods/pickle-tweaks) | BlakeBr0 | 59 | MIT | 宽松许可 | [仓库](https://github.com/BlakeBr0/PickleTweaks) |
| [Refined Storage](https://www.curseforge.com/minecraft/mc-mods/refined-storage) | raoulvdberge | 58 | MIT | 宽松许可 | [仓库](https://github.com/refinedmods/refinedstorage2) |
| [Peruvian's Delight](https://www.curseforge.com/minecraft/mc-mods/peruvians-delight) | GiulianoMMPP | 57 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/GiulianoMaPe/PeruviansDelight-1.20.1-Forge) |
| [Arbitrary Delight](https://www.curseforge.com/minecraft/mc-mods/arbitrary-delight) | Ri0t69 | 57 | MIT | 宽松许可 | [仓库](https://github.com/VaporRi0t/Arbitrary-Delight) |
| [Advanced Rocketry](https://www.curseforge.com/minecraft/mc-mods/advanced-rocketry) | zmaster587 | 56 | MIT | 宽松许可 | [仓库](https://github.com/Advanced-Rocketry/AdvancedRocketry) |
| [Eternal Starlight's Delight](https://modrinth.com/mod/eternal-starlight-delight) | Minor2CCh | 56 | MIT | 宽松许可 | [仓库](https://github.com/Minor2CCh/EternalStarlightDelight) |
| [Mining Gadgets](https://www.curseforge.com/minecraft/mc-mods/mining-gadgets) | Direwolf20 | 54 | MIT | 宽松许可 | [仓库](https://github.com/Direwolf20-MC/MiningGadgets) |
| [Simply Jetpacks 2](https://www.curseforge.com/minecraft/mc-mods/simply-jetpacks-2) | Tomson124 | 54 | MIT | 宽松许可 | [仓库](https://github.com/Tomson124/SimplyJetpacks2) |
| [Project Red - Core](https://www.curseforge.com/minecraft/mc-mods/project-red-core) | MrTJP | 53 | MIT | 宽松许可 | [仓库](https://github.com/MrTJP/ProjectRed) |
| [bodacious berries](https://www.curseforge.com/minecraft/mc-mods/bodacious-berries) | eristhea | 52 | MIT | 宽松许可 | [仓库](https://github.com/rosebudmods/bodacious-berries) |
| [Silent's Gems](https://www.curseforge.com/minecraft/mc-mods/silents-gems) | SilentChaos512 | 51 | MIT | 宽松许可 | [仓库](https://github.com/SilentChaos512/SilentGems) |
| [Infernal Expansion](https://www.curseforge.com/minecraft/mc-mods/infernal-expansion) | InfernalStudios | 50 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/infernalstudios/Infernal-Expansion) |
| [Spartan Shields](https://www.curseforge.com/minecraft/mc-mods/spartan-shields) | ObliviousSpartan | 48 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/ObliviousSpartan/SpartanShields) |
| [Extended Industrialization](https://www.curseforge.com/minecraft/mc-mods/extended-industrialization) | swedzish | 48 | MIT | 宽松许可 | [仓库](https://github.com/Swedz/Extended-Industrialization) |
| [Ramadan Delight (A Farmer's Delight Add-on)](https://www.curseforge.com/minecraft/mc-mods/ramadan-delight) | Eggmain | 48 | MIT | 宽松许可 | [仓库](https://github.com/MehdiNoui/RamadanDelight) |
| [Vanilla Delights](https://www.curseforge.com/minecraft/mc-mods/vanilla-delight) | Chaolux | 48 | MIT | 宽松许可 | [仓库](https://github.com/Chaolux1/VanillaDelight) |
| [Roots Classic](https://www.curseforge.com/minecraft/mc-mods/roots-classic) | Lothrazar | 47 | MIT | 宽松许可 | [仓库](https://github.com/Lothrazar/RootsClassic) |
| [Avaritia's Delight](https://www.curseforge.com/minecraft/mc-mods/avaritias-delight) | IAFEnvoy | 47 | MIT | 宽松许可 | [仓库](https://github.com/Nova-Committee/AvaritiaDelight) |
| [ Candy World - ReCaramelized](https://www.curseforge.com/minecraft/mc-mods/candy-world-recaramelized) | Mrbysco | 46 | MIT | 宽松许可 | [仓库](https://github.com/Mrbysco/CandyWorld) |
| [Create Big Cannons](https://www.curseforge.com/minecraft/mc-mods/create-big-cannons) | rbasamoyai | 45 | 见复核说明 | 仅限非商用 | [仓库](https://github.com/Cannoneers-of-Create/CreateBigCannons) |
| [Jaden's Nether Expansion Delight](https://www.curseforge.com/minecraft/mc-mods/jadens-nether-expansion-delight) | Chaolux | 43 | MIT | 宽松许可 | [仓库](https://github.com/Chaolux1/NetherExpansionDelight) |
| [Millet Delight](https://modrinth.com/mod/millet-delight) | DreaMemory2 | 43 | MIT | 宽松许可 | [仓库](https://github.com/DreaMemory2/MilletDelight) |
| [SpoiledZ](https://modrinth.com/mod/spoiledz) | Globox_Z | 42 | MIT | 宽松许可 | [仓库](https://github.com/Globox1997/SpoiledZ) |
| [Extended Crafting](https://www.curseforge.com/minecraft/mc-mods/extended-crafting) | BlakeBr0 | 41 | MIT | 宽松许可 | [仓库](https://github.com/BlakeBr0/ExtendedCrafting) |
| [Armor of the Ages](https://www.curseforge.com/minecraft/mc-mods/armor-of-the-ages) | dawnoftime_team | 41 | MIT | 宽松许可 | [仓库](https://github.com/Dawn-of-Time-Project/armoroftheages) |
| [Cognition (Experience Obelisk)](https://www.curseforge.com/minecraft/mc-mods/cognition-mod) | cyanog3n | 40 | 见复核说明 | 宽松许可 | [仓库](https://github.com/terrarium-earth/Cognition) |
| [More Delight (for Farmer's Delight)](https://www.curseforge.com/minecraft/mc-mods/more-delight-forge) | axperty | 40 | MIT | 宽松许可 | [仓库](https://github.com/axperty/moredelight) |
| [Mystical Agradditions](https://www.curseforge.com/minecraft/mc-mods/mystical-agradditions) | BlakeBr0 | 39 | MIT | 宽松许可 | [仓库](https://github.com/BlakeBr0/MysticalAgradditions) |
| [Dumplings Delight Rewrapped](https://www.curseforge.com/minecraft/mc-mods/dumplings-delight-rewrapped) | Joshcraft2002 | 39 | MIT | 宽松许可 | [仓库](https://github.com/Joshcraft2002/Dumplings-Delight-Rewrapped) |
| [Muffin's Thai's Delight](https://modrinth.com/mod/muffins-thais-delight) | firemuffin303 | 38 | MIT | 宽松许可 | [仓库](https://github.com/FireMuffin303/Muffins-ThaiDelight) |
| [No Tree Punching](https://www.curseforge.com/minecraft/mc-mods/no-tree-punching) | AlcatrazEscapee | 37 | MIT | 宽松许可 | [仓库](https://github.com/alcatrazEscapee/no-tree-punching) |
| [Farmer's Delight: Plus](https://modrinth.com/mod/farmers-delight-plus) | Johnyele | 37 | MIT | 宽松许可 | [仓库](https://github.com/Johnyel/FarmersDelightPlus) |
| [JAOPCA](https://www.curseforge.com/minecraft/mc-mods/jaopca) | TheLMiffy1111 | 34 | MIT | 宽松许可 | [仓库](https://github.com/TheLMiffy1111/JAOPCA) |
| [LaserIO](https://www.curseforge.com/minecraft/mc-mods/laserio) | Direwolf20 | 33 | MIT | 宽松许可 | [仓库](https://github.com/Direwolf20-MC/LaserIO) |
| [Exposure](https://www.curseforge.com/minecraft/mc-mods/exposure) | mortuusars | 32 | MIT | 宽松许可 | [仓库](https://github.com/mortuusars/Exposure) |
| [Chalk (Fabric)](https://www.curseforge.com/minecraft/mc-mods/chalk-fabric) | DaFuqsy | 32 | MIT | 宽松许可 | [仓库](https://github.com/DaFuqs/Chalk) |
| [Things [Fabric]](https://www.curseforge.com/minecraft/mc-mods/things-fabric) | gliscowo | 31 | MIT | 宽松许可 | [仓库](https://github.com/wisp-forest/things) |
| [Fairy Lights](https://www.curseforge.com/minecraft/mc-mods/fairy-lights) | pau101 | 30 | MIT | 宽松许可 | [仓库](https://github.com/pau101/Fairy-Lights) |
| [Barbeque's Delight [Forge/NeoForge]](https://www.curseforge.com/minecraft/mc-mods/barbeques-delight) | lcy0x1 | 30 | MIT | 宽松许可 | [仓库](https://github.com/Minecraft-LightLand/BarbequesDelight) |
| [DungeonZ](https://www.curseforge.com/minecraft/mc-mods/dungeonz) | Globox_Z | 29 | MIT | 宽松许可 | [仓库](https://github.com/Globox1997/DungeonZ) |
| [Simply Tea!](https://www.curseforge.com/minecraft/mc-mods/simply-tea) | EpicSquid315 | 29 | MIT | 宽松许可 | [仓库](https://github.com/KnightMiner/SimplyTea) |
| [Christmas Delight](https://modrinth.com/mod/christmas-delight) | ThePoupyBuTers | 29 | MIT | 宽松许可 | [仓库](https://github.com/Cat-Plays/Christmas_Delight-Fabric_1.21) |
| [Veggie Delight](https://modrinth.com/mod/veggie-delight) | ElectroArtifex | 28 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/ElectroArtifex/veggiedelight-1.20.1) |
| [Iron Furnaces](https://www.curseforge.com/minecraft/mc-mods/iron-furnaces) | XenoMustache | 27 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/Qelifern/IronFurnacesNeoForge) |
| [Extended Mushrooms](https://www.curseforge.com/minecraft/mc-mods/extended-mushrooms) | thecech12 | 27 | MIT | 宽松许可 | [仓库](https://github.com/cech12/ExtendedMushrooms) |
| [Tea Aroma](https://www.curseforge.com/minecraft/mc-mods/tea-aroma) | foggyhillside | 27 | MIT | 宽松许可 | [仓库](https://github.com/FoggyHillside/Tea-Aroma) |
| [Dawn of Time](https://www.curseforge.com/minecraft/mc-mods/dawn-of-time) | dawnoftime_team | 26 | MIT | 宽松许可 | [仓库](https://github.com/Dawn-of-Time-Project/dawnoftime) |
| [Hot Kettles](https://www.curseforge.com/minecraft/mc-mods/hotkettles) | slainlight | 26 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/matthewperiut/hotkettles) |
| [Legume Delight](https://www.curseforge.com/minecraft/mc-mods/legume-delight) | pawjwp | 26 | CC BY-NC-SA 4.0 | 仅限非商用 | [仓库](https://github.com/pawjwp/Legume-Delight) |
| [Building Gadgets](https://www.curseforge.com/minecraft/mc-mods/building-gadgets) | Direwolf20 | 25 | MIT | 宽松许可 | [仓库](https://github.com/Direwolf20-MC/BuildingGadgets) |
| [Ocean's Delight](https://www.curseforge.com/minecraft/mc-mods/oceans-delight) | scouter567 | 25 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/Scouter456/Oceans_Delight) |
| [Totemic](https://www.curseforge.com/minecraft/mc-mods/totemic) | ljfa2 | 25 | MIT | 宽松许可 | [仓库](https://github.com/TeamTotemic/Totemic) |
| [Corn Delight](https://www.curseforge.com/minecraft/mc-mods/corn-delight) | SyameimaruZheng | 25 | MIT | 宽松许可 | [仓库](https://github.com/0999312/Corn-Delight) |
| [Nether's Delight](https://www.curseforge.com/minecraft/mc-mods/nethers-delight) | lumpazl | 23 | MIT | 宽松许可 | [仓库](https://github.com/Chefs-Delight/NethersDelight_Forge) |
| [Dynamic Trees](https://www.curseforge.com/minecraft/mc-mods/dynamictrees) | MaxHyper | 23 | MIT | 宽松许可 | [仓库](https://github.com/DynamicTreesTeam/DynamicTrees) |
| [Tarot Cards](https://www.curseforge.com/minecraft/mc-mods/tarot-cards) | shiroroku | 23 | MIT | 宽松许可 | [仓库](https://github.com/shiroroku/TarotCards) |
| [Angling](https://modrinth.com/mod/angling) | EightSidedSquare | 23 | CC0 1.0 | 许可明确覆盖素材 | [仓库](https://github.com/8s2/Angling) |
| [AgriCraft](https://www.curseforge.com/minecraft/mc-mods/agricraft) | InfinityRaider | 22 | MIT | 宽松许可 | [仓库](https://github.com/AgriCraft/AgriCraft) |
| [Integrated Dynamics](https://www.curseforge.com/minecraft/mc-mods/integrated-dynamics) | kroeser | 21 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/CyclopsMC/IntegratedDynamics) |
| [Construction Sticks](https://www.curseforge.com/minecraft/mc-mods/construction-sticks) | Mrbysco | 21 | MIT | 宽松许可 | [仓库](https://github.com/Mrbysco/ConstructionSticks) |
| [Vietnam's Delight Reborn](https://modrinth.com/mod/vietnams-delight-reborn) |  | 21 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/MikeStorm03/vietnams-delight) |
| [Easter's Delight](https://www.curseforge.com/minecraft/mc-mods/easters-delight) | PhantomWing | 20 | MIT | 宽松许可 | [仓库](https://github.com/PhantomWing/EasterDelight) |
| [realisticdining](https://www.curseforge.com/minecraft/mc-mods/realisticdining) | EEWW245 | 20 | MIT | 宽松许可 | [仓库](https://github.com/EEWW245/realisticdining) |
| [Ceramics](https://www.curseforge.com/minecraft/mc-mods/ceramics) | KnightMiner | 19 | MIT | 宽松许可 | [仓库](https://github.com/KnightMiner/Ceramics) |
| [Copper Pot](https://www.curseforge.com/minecraft/mc-mods/copper-pot) | davigj | 19 | MIT | 宽松许可 | [仓库](https://github.com/davidjjung/copperpot) |
| [RFTools Control](https://www.curseforge.com/minecraft/mc-mods/rftools-control) | McJty | 18 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsControl) |
| [Wraith Waystones](https://www.curseforge.com/minecraft/mc-mods/fabric-waystones) | LordDeatHunter | 18 | MIT | 宽松许可 | [仓库](https://github.com/LordDeatHunter/FabricWaystones) |
| [Chalk](https://www.curseforge.com/minecraft/mc-mods/chalk) | mortuusars | 18 | MIT | 宽松许可 | [仓库](https://github.com/mortuusars/Chalk) |
| [Large Meals](https://modrinth.com/mod/large-meals) | KreloX | 18 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/KreloX/Large-Meals) |
| [Functional Storage](https://www.curseforge.com/minecraft/mc-mods/functional-storage) | Buuz135 | 17 | MIT | 宽松许可 | [仓库](https://github.com/Buuz135/FunctionalStorage) |
| [Eden Ring Reforked](https://www.curseforge.com/minecraft/mc-mods/eden-ring-reforked) | ErzeKawek | 17 | MIT | 宽松许可 | [仓库](https://github.com/ErzeKawek/EdenRing-Reforked) |
| [Spanish Delight](https://modrinth.com/mod/spanish-delight) | DevJulen | 17 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/DevJulen/SpanishDelight-Forge) |
| [Numismatic Overhaul](https://www.curseforge.com/minecraft/mc-mods/numismatic-overhaul) | gliscowo | 16 | MIT | 宽松许可 | [仓库](https://github.com/wisp-forest/numismatic-overhaul) |
| [Beans Backpacks (Legacy)](https://www.curseforge.com/minecraft/mc-mods/beans-backpacks) | BeansGalaxy | 16 | MIT | 宽松许可 | [仓库](https://github.com/BeansGalaxy/Beans-Backpacks-2) |
| [Vegehenna: A vegetarian's heaven?](https://modrinth.com/mod/vegehenna) | ivangeevo | 16 | CC BY 4.0 | 许可明确覆盖素材 | [仓库](https://github.com/ivangeevo/vegehenna) |
| [Maple Delight](https://www.curseforge.com/minecraft/mc-mods/maple-delight) | HoldWind | 16 | MIT | 宽松许可 | [仓库](https://github.com/HLHT-Studios/maple-delight) |
| [Habitat](https://www.curseforge.com/minecraft/mc-mods/habitat) | schnappdragon | 15 | MIT | 宽松许可 | [仓库](https://github.com/SnappyDragon64/Habitat) |
| [RFTools Builder](https://www.curseforge.com/minecraft/mc-mods/rftools-builder) | McJty | 14 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsBuilder) |
| [Not Enough Wands](https://www.curseforge.com/minecraft/mc-mods/not-enough-wands) | romelo333 | 14 | MIT | 宽松许可 | [仓库](https://github.com/romelo333/notenoughwands1.8.8) |
| [Fungi Delight (A Farmer's Delight Add-on)](https://www.curseforge.com/minecraft/mc-mods/fungi-delight) | Eggmain | 14 | MIT | 宽松许可 | [仓库](https://github.com/MehdiNoui/FungiDelight) |
| [Better Archeology](https://www.curseforge.com/minecraft/mc-mods/better-archeology) | ruven | 13 | MIT | 宽松许可 | [仓库](https://github.com/Pandarix/BetterArcheology-MultiLoaded) |
| [Nears](https://modrinth.com/mod/nears) | digitalpear | 13 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/DigitalPear/Nears-Fabric) |
| [Advanced Peripherals](https://www.curseforge.com/minecraft/mc-mods/advanced-peripherals) | srrendi | 12 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/IntelligenceModding/AdvancedPeripherals) |
| [Building Wands](https://www.curseforge.com/minecraft/mc-mods/building-wands) | nicguzzo | 12 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/nicguzzo/wands) |
| [Extreme Reactors](https://www.curseforge.com/minecraft/mc-mods/extreme-reactors) | ZeroNoRyouki | 11 | MIT | 宽松许可（证据较弱） | [仓库](https://github.com/ZeroNoRyouki/ExtremeReactors2) |
| [Hostile Neural Networks](https://www.curseforge.com/minecraft/mc-mods/hostile-neural-networks) | Shadows_of_Fire | 11 | MIT | 宽松许可 | [仓库](https://github.com/Shadows-of-Fire/Hostile-Neural-Networks) |
| [Scannable](https://www.curseforge.com/minecraft/mc-mods/scannable) | Sangar | 11 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/MightyPirates/Scannable) |
| [Kobolds!](https://www.curseforge.com/minecraft/mc-mods/kobolds) | Jusey1 | 11 | MIT | 宽松许可 | [仓库](https://github.com/Jusey1/kobolds) |
| [Farmer's Delight Expansion](https://modrinth.com/mod/farmers-delight-expansion) | keishispl | 11 | MIT | 宽松许可 | [仓库](https://github.com/keishispl/farmers-delight-expansion) |
| [Farmer's Crops](https://modrinth.com/mod/farmers-crops) | JCS | 11 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/JCS-Mecabricks/Farmers-Crops) |
| [Construction Wand](https://www.curseforge.com/minecraft/mc-mods/construction-wand) | ThetaDev | 10 | MIT | 宽松许可 | [仓库](https://github.com/Theta-Dev/ConstructionWand) |
| [Scaling Health](https://www.curseforge.com/minecraft/mc-mods/scaling-health) | SilentChaos512 | 10 | MIT | 宽松许可 | [仓库](https://github.com/SilentChaos512/ScalingHealth) |
| [Yoyos](https://www.curseforge.com/minecraft/mc-mods/yoyos) | jozufozu | 10 | MIT | 宽松许可 | [仓库](https://github.com/Jozufozu/Yoyos) |
| [Iron Jetpacks](https://www.curseforge.com/minecraft/mc-mods/iron-jetpacks) | BlakeBr0 | 9 | MIT | 宽松许可 | [仓库](https://github.com/BlakeBr0/IronJetpacks) |
| [RFTools Utility](https://www.curseforge.com/minecraft/mc-mods/rftools-utility) | McJty | 9 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsUtility) |
| [Useful Backpacks](https://www.curseforge.com/minecraft/mc-mods/useful-backpacks) | HyCraftHD | 8 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/MC-U-Team/Useful-Backpacks) |
| [Tinkers' Levelling Addon](https://www.curseforge.com/minecraft/mc-mods/tinkers-levelling-addon) | Pyre540 | 8 | MIT | 许可明确覆盖素材 | [仓库](https://github.com/Pyre540/tinkers-levelling-addon) |
| [Tinkers' Mechworks](https://www.curseforge.com/minecraft/mc-mods/tinkers-mechworks) | mDiyo | 7 | CC BY 3.0 | 许可明确覆盖素材 | [仓库](https://github.com/SlimeKnights/TinkersMechworks) |
| [Compact Machines](https://www.curseforge.com/minecraft/mc-mods/compact-machines) | davenonymous | 6 | MIT | 宽松许可 | [仓库](https://github.com/CompactMods/CompactMachines) |
| [Bygone Nether](https://www.curseforge.com/minecraft/mc-mods/bygone-nether) | izofar | 6 | MIT | 宽松许可 | [仓库](https://github.com/izofar/bygone-nether) |
| [RFTools Power](https://www.curseforge.com/minecraft/mc-mods/rftools-power) | McJty | 5 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsPower) |
| [Tom's Simple Storage Mod](https://www.curseforge.com/minecraft/mc-mods/toms-storage) | tom54541 | 5 | MIT | 宽松许可 | [仓库](https://github.com/tom5454/Toms-Storage) |
| [More Red](https://www.curseforge.com/minecraft/mc-mods/more-red) | Commoble | 5 | MIT | 宽松许可 | [仓库](https://github.com/Commoble/morered) |
| [Wolf Armor and Storage Legacy](https://www.curseforge.com/minecraft/mc-mods/wolf-armor-and-storage) | saturniidae | 5 | MIT | 宽松许可 | [仓库](https://github.com/Smallinger/wolf-armor-and-storage-legacy) |
| [Project Red - Fabrication](https://www.curseforge.com/minecraft/mc-mods/project-red-fabrication) | MrTJP | 5 | MIT | 宽松许可 | [仓库](https://github.com/MrTJP/ProjectRed) |
| [Mekanism Generators](https://www.curseforge.com/minecraft/mc-mods/mekanism-generators) | bradyaidanc | 4 | MIT | 宽松许可 | [仓库](https://github.com/mekanism/Mekanism) |
| [Ender Storage 1.8.+](https://www.curseforge.com/minecraft/mc-mods/ender-storage-1-8) | covers1624 | 4 | MIT | 宽松许可 | [仓库](https://github.com/TheCBProject/EnderStorage) |
| [RFTools Storage](https://www.curseforge.com/minecraft/mc-mods/rftools-storage) | McJty | 3 | MIT | 宽松许可 | [仓库](https://github.com/McJtyMods/RFToolsStorage) |
| [Chickens](https://www.curseforge.com/minecraft/mc-mods/chickens) | setycz | 3 | MIT | 宽松许可 | [仓库](https://github.com/setycz/ChickensMod) |
| [Valleyman's bean](https://www.curseforge.com/minecraft/mc-mods/valleymans-beans) | thejurassicking1 | 3 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/Floi1/Valleymans-Beans-mod) |
| [Potion Rings](https://www.curseforge.com/minecraft/mc-mods/potion-fingers-redux) | Shivaxi | 1 | WTFPL | 宽松许可 | [仓库](https://github.com/Duck-XYZ/potion_rings) |
| [Backpacks!](https://www.curseforge.com/minecraft/mc-mods/backpacks) | Brad16840 | 1 | Apache-2.0 | 宽松许可 | [仓库](https://github.com/Eclipse-Studios/backpacks) |

## 合成贴图

合成贴图不取自任何模组，是按 4,046 个日常物品的名称清单（`synth/concepts/`）生成的。

| 数据集 | 贴图数 | 物品数 | 生成方式 |
|---|---|---|---|
| synth2 | 6,922 | 3,521 | 本项目的 LoRA 直接生成，每个物品 2～3 张 |
| many | 44,011 | 3,846 | 本项目的 LoRA 直接生成，每个物品约 20 张，筛选后保留 |
| manual | 27,413 | 1,306 | 外部图像生成模型出底稿，再由本项目的 LoRA 重画 |

- “本项目的 LoRA”是在 Stable Diffusion XL 上、只用上面列出的手绘贴图训练的风格 LoRA，它让 SDXL 直接画出 16×16 网格的贴图。
- 外部图像生成模型只用于 1,306 个 LoRA 画不好的物品，每个物品 3 张底稿，共 3,918 张；底稿缩成 16×16 后进入训练集，并由 LoRA 以它为底各重画成多张变体。
- 贴图的筛选用了 CLIP（判断生成的图是否像它的名字）。

## 用到的模型和组件

| 名称 | 用途 | 许可 |
|---|---|---|
| [Stable Diffusion XL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 生成合成贴图（仅训练阶段） | CreativeML Open RAIL++-M |
| [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | 同上 | MIT |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | 筛选合成贴图（仅训练阶段） | MIT |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 文本编码器，随模型一起分发 | MIT |
| [U²-Net-p](https://github.com/xuebinqin/U-2-Net) | 从参考照片里抠出物体，随客户端和模型一起分发（ONNX 文件取自 [rembg](https://github.com/danielgatis/rembg) 的发布页，原样分发） | [Apache-2.0](licenses/Apache-2.0.txt) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 客户端推理 | MIT |
