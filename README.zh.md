# StarItem

中文 | [English](README.md)

输入一个物品名，生成 Minecraft 风格的 16×16 物品贴图。每次出四张，挑喜欢的保存成 PNG。

![客户端截图](docs/screenshot.png)

- 一个 exe，双击就能用，不需要安装 Python 或显卡驱动，模型已经内置。
- 中文、英文都可以输入：`红宝石镐`、`golden crown`、`闹钟`、`潜水艇`。
- 不需要显卡，在普通 CPU 上出四张图只要几秒。
- 输出是带透明背景的 16×16 PNG，可以直接放进资源包或模组。

## 使用

1. 从发布页下载 `StarItem_SIC-1.2.exe`（约 605 MB），放在任意位置运行。目前只有 Windows x64 版本。
2. 在输入框里写物品名或一句简短的描述，点“生成”。
3. 每张图下面有“保存 PNG”，结果区右上角的“全部保存到文件夹”可以一次保存四张。

界面语言跟随系统，也可以用标题栏上的按钮切换。界面上的几个选项：

| 选项 | 作用 |
|---|---|
| 引导强度 | 越高越贴近描述，越低几张图之间的变化越多。默认 5.5。 |
| 步数 | 采样步数，默认 30。调高会变慢，画质提升有限。 |
| 随机种子 | 取消勾选后可以填固定的种子，同样的输入会得到同样的图。 |
| 名称补全 | 输入常见叫法时自动补成模型训练时见过的完整名称，例如“闹钟”会补成“闹钟, alarm clock”。状态栏会显示补全后的内容。 |
| 画风 | “手绘”偏向模组贴图的画法，“合成”偏向日常物品的画法。勾选“自动”时由客户端按物品决定，一般不用动。 |
| 网络 | 模型里有两个出图网络，叫 Mo 和 Jang。Jang 认得名字的物品画得准，Mo 更听颜色词、对没见过的组合更稳。“自动”时，提示词恰好是模型认得的名字就全交给 Jang，其余情况两个各出两张，每张图下面标着是谁画的；觉得其中一个更合适，可以手动只用它。 |
| 参考图 | 可选。选一张物体的图片或照片（PNG 或 JPG，画面里一个物体），或者直接把图片拖进窗口，生成的贴图会照着它的形状和颜色画。纯色或透明的背景会直接去掉，实拍照片会自动抠出主体；参考图一栏的小缩略图就是抠出来的结果。物品名仍然要写。“变化幅度”越低越贴近参考图，越高画得越干净；“跟随颜色”控制结果的颜色向参考图靠多少。 |

写提示词的建议：

- 直接写物品名效果最好，模型不擅长理解长句。
- 可以在名字前加颜色或材质：`紫色的剑`、`golden crown`。
- 结果不理想时多点几次“生成”，或者换一种叫法（“电脑”和“笔记本电脑”的结果可能不同）。
- 手头有实物的图片时，用参考图通常比反复改提示词更有效。模型认不出名字的东西（例如只写“望远镜”，它分不清是单筒、双筒还是天文望远镜），给一张图往往就能画对。

## 模型：SIC-1.2

| | |
|---|---|
| 类型 | 像素空间的扩散模型，直接生成 16×16 RGBA 图像 |
| 网络 | 两个 UNet：Mo 约 6,070 万参数，Jang 约 6,430 万参数（多出的是每个词一组的可训练向量） |
| 文本编码器 | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)（冻结，量化后随模型分发） |
| 采样 | DDIM，默认 30 步，classifier-free guidance |
| 格式 | ONNX，用 ONNX Runtime 在 CPU 上推理 |
| 大小 | 模型文件共约 570 MB，另有 4.6 MB 的抠图模型 |

两个网络用同一份数据训练，共用分词器、文本编码器和名称补全表。区别在文字条件：Jang 在冻结的文本特征上给每个词加了一组可训练的向量，能把意思相近的物品按字面分开，所以认名字准得多；代价是遇到没见过的组合时不如 Mo 稳。按自动评估，日常物品清单里“画出来最像的就是这个物品”的比例，Mo 是 34%，Jang 是 57%；给物品名前面加一个颜色词时，Jang 更保得住物品，Mo 更听颜色。所以客户端按提示词分配，拿不准的时候两个都出。

训练数据共约 9.6 万张 16×16 贴图：

| 来源 | 贴图数 | 说明 |
|---|---|---|
| 开源模组的手绘物品贴图 | 17,567 | 来自 200 个模组，素材许可逐个复核过 |
| 合成贴图 | 78,346 | 按 4,046 个日常物品的名称清单，用图像生成模型生成并筛选 |

完整的来源、作者和许可见 [DATA_SOURCES.md](DATA_SOURCES.md)。

## 给 AI 助手用

模型文件也单独发布（`manifest.json`、`unet.onnx`、`unet_names.onnx`、`text_encoder.onnx`、`tokenizer.json`、`aliases.json`、`known_names.json`、`coverage.json`），不用客户端也能调用。`unet.onnx` 是 Mo，`unet_names.onnx` 是 Jang；只认识一个网络的旧版客户端读这个模型包时会只用 Mo。

`skill/staritem-sprites/` 是一个给文本 AI 助手用的技能：助手按物品名调用模型出图，自己看预览图挑一张，再放进你的模组目录。它只有一个说明文件和一个独立的 Python 脚本，需要 `onnxruntime`、`numpy`、`tokenizers` 三个包。脚本还可以照着一张参考图或照片来画（`--ref 图片`，另需 `pillow`）。

脚本也可以直接在命令行用：

```bash
python skill/staritem-sprites/scripts/generate.py "闹钟" "红宝石镐" --models <模型目录> --out sprites
```

## 已知的局限

- 只能生成 16×16 的贴图。
- 模组里常见的物品（工具、武器、锭、药水、食物）效果最稳定。日常物品清单里的 4,046 个物品，按自动评估大约四分之三能稳定画对；其余的可能只能画出“大致是这一类东西”。
- 结构复杂或很细的物体在 16×16 下本来就难表现，例如相机、纺车、拼图。
- 没见过的“材质 + 物品”组合是抽签：Jang 颜色跟得准，但法杖、魔杖配生僻材质时容易在杖头画出一块色块；Mo 的形状更稳，但颜色常常不对。两个各出两张就是为了这个。
- 模型不理解复杂的描述，例如“左边有一颗星星的蓝色盾牌”。
- 实拍照片的抠图靠一个找“主体”的小模型：画面里只有一个突出的物体时效果好；有好几样东西、或者物体占满整个画面时，可能抠错或抠不出来，这时先把图裁到只剩那个物体再试。
- 商标、文字这类细节在 16×16 里留不住，参考图也帮不上。
- 生成的贴图可能与训练集中的某张贴图相似，用于公开发布的作品前请自行判断。

## 许可

- **代码**：[MIT 许可](LICENSE)。
- **抠图模型**：参考照片的抠图用的是 U²-Net-p，[Apache-2.0 许可](licenses/Apache-2.0.txt)，原样分发；客户端状态栏的“许可”里可以看到全文。
- **模型权重**：[CC BY-NC-SA 4.0](LICENSE-MODEL.md)。训练数据里包含 CC BY-NC、CC BY-NC-SA 许可的素材，所以权重只能用于非商业用途。发布的 exe 内置了权重，因此 exe 整体也只能用于非商业用途。
- 素材来源与署名见 [DATA_SOURCES.md](DATA_SOURCES.md)。如果你是其中某个模组的作者并希望移除自己的素材，请提 issue。
- 本项目不是 Minecraft 官方产品，与 Mojang 和 Microsoft 没有关联，也未获得它们的认可。Mo 和 Jang 只是两个网络的昵称。

## 从源码构建

客户端在 `client/` 下，用 Rust 编写（egui + ONNX Runtime）：

```bash
cd client
cargo build --release
```

模型文件放在 `client/models/` 下，编译时会打包进 exe。也可以在运行时用 `--models <目录>` 指定另一套模型。

## 训练流程

训练用 Python，脚本分三组：

| 目录 | 内容 |
|---|---|
| `tools/` | 模组素材许可的复核、下载、解包、清洗和标注 |
| `synth/` | 合成贴图：训练风格 LoRA、按物品清单生成、筛选、导入外部底稿并重画 |
| `train/` | 训练扩散模型、评估、导出 ONNX |

大致步骤：

```bash
# 1. 整理手绘贴图
python -X utf8 tools/extract_assets.py
python -X utf8 tools/clean_and_label.py

# 2. 合成贴图
python -X utf8 synth/train_lora.py --steps 6000
python -X utf8 synth/generate_lora.py --per 20
python -X utf8 synth/build_dataset.py --raw dataset/many_raw --out dataset/many --direct --all-concepts --max-rank 300

# 3. 训练并导出
python -X utf8 train/train.py --name sic16 --ch 128 --extra dataset/synth2 dataset/many dataset/manual --steps 150000
python -X utf8 train/train.py --name sic17 --ch 128 --extra dataset/synth2 dataset/many dataset/manual --steps 150000 --tok-emb --cap-aug-synth 0
python -X utf8 train/export_onnx.py --run sic16 --names-run sic17 --name SIC-1.2
```

各脚本开头的说明里有完整的参数和用法。

## 致谢

感谢所有以开放许可发布素材的模组作者，名单见 [DATA_SOURCES.md](DATA_SOURCES.md)。
