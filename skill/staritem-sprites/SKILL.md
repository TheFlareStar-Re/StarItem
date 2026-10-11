---
name: staritem-sprites
description: Generate Minecraft-style 16x16 pixel-art item sprites (PNG with transparent background) from an item name, using the local StarItem SIC model. Use this whenever the user wants an item texture or icon for a Minecraft mod, resource pack or data pack, asks for a 16x16 pixel-art item sprite, needs placeholder art for items in a mod they are writing, or asks to "draw", "make" or "generate" a Minecraft item such as a sword, ingot, food, tool or everyday object, in Chinese or English, even if they do not mention StarItem. Also use it when they have a picture of an object and want it turned into such a sprite. Do not use it for block textures, entity skins, GUI art, or images larger than 16x16.
---

# StarItem sprites

StarItem's SIC model turns an item name into Minecraft-style 16×16 item sprites. It is a small diffusion model that runs on the CPU in a few seconds, so generating several candidates and picking the best one is cheap. This skill wraps it in one script.

## Setup

The script needs Python with `onnxruntime`, `numpy` and `tokenizers` (plus `pillow` for reference pictures), and the model folder, which is released separately from this skill. The folder holds `manifest.json`, `unet.onnx`, `unet_names.onnx`, `text_encoder.onnx`, `tokenizer.json`, `aliases.json`, `known_names.json` and `coverage.json`, plus `matting.onnx`, a small model that is only needed to cut the object out of a reference photo.

The script looks for the model in this order: `--models <folder>`, the `STARITEM_MODELS` environment variable, `./models`, `~/.staritem/models`, and a `models` folder inside this skill. Every mode except `--preview-of` needs it. If the model or a package is missing, the script prints a JSON line with an `error` field that says exactly what is missing. Relay that to the user rather than guessing a location or downloading something on your own; where the model lives is their decision.

## Generating

```bash
python scripts/generate.py "alarm clock" --out sprites
python scripts/generate.py "红宝石镐" "golden crown" "潜水艇" --out sprites --count 6
```

For each prompt the script writes:

- `<out>/<prompt>_<seed>_1.png` … `_N.png`: the sprites, 16×16 RGBA with a transparent background. These are the deliverables. The file name is the prompt with spaces turned into underscores, so a Chinese prompt gives a Chinese file name.
- `<out>/<prompt>_<seed>_preview.png`: the same sprites side by side, enlarged on a checkerboard. This one is for looking at.

and prints one JSON line:

```json
{"prompt": "alarm clock", "sent_to_model": "闹钟, alarm clock", "name_completed": true, "training_sprites": 22, "style": 1.0, "seed": 5, "model": "SIC-1.2", "files": ["..."], "preview": "...", "warnings": [], "route": "everyday_object", "networks": ["jang", "jang", "jang", "jang"]}
```

- `training_sprites` is how many training images the model had for the everyday object it matched. Nearly every listed object has ten or more, so a high number is not a promise: `相机` maps to a name with 35 and still comes out as a video camera. A number under ten does mean the model barely knows the object; expect poor results and go straight to "When the model does not know the item" below. It is `null` when no everyday object was matched, which is normal for mod-style items.
- `warnings` names candidates that are nearly empty or have no transparent background; skip those.
- `route` and `networks` say which of the model's two networks drew each candidate; see "Two networks" below.

Options:

- `--count N`: candidates per prompt, 1–16, default 4. Four to eight is a good working size; the preview shrinks its enlargement automatically as the count grows.
- `--seed N`: makes a run reproducible. Candidate k of a seed is the same image whatever the count, so `--seed S --count k` brings back any earlier candidate. With several prompts, each gets the next seed. Without `--seed`, every prompt gets its own random seed, reported in the JSON.
- `--steps`, `--guidance`: the model's defaults (30 and 5.5) are right almost always.
- `--style`, `--no-expand`, `--network`: see below.

## Two networks

From SIC-1.2 on the model folder holds two networks, nicknamed Mo and Jang, and the script decides per prompt which one draws:

- **Jang** knows items by name. Ask for a listed everyday object or for an item that exists in the mods it was trained on, and it draws that item far more reliably than Mo. It also gets materials and colors of fantasy items right (`翡翠镐` is green). Its weak spot is combinations it never saw: staffs and wands with unusual materials (`闪电法杖`, `翡翠魔杖`) often come out with a stray block of color at the head.
- **Mo** is steadier on such new combinations and follows a color word in front of an everyday object more readily (`蓝色的雨伞`), but it confuses everyday objects with similar ones (an owl becomes some white bird) and often picks the wrong color for a material.

`route` in the JSON says what the script did:

- `everyday_object` or `known_item`: the prompt is exactly a name the model knows, so every candidate comes from Jang.
- `mixed`: a name with extra words in front, or a combination nobody drew. The candidates alternate, Mo first, and `networks` lists who drew which. Neither is reliably better here, so compare them in the preview.
- `forced`: you chose with `--network`.

When one network's candidates are clearly better for a prompt, rerun with `--network mo` or `--network jang` and a larger `--count` instead of taking half the candidates from the weaker one. Candidate k of a seed uses the same starting noise whichever network draws it, so the same seed with the other network shows the same compositions drawn differently.

## Look before you deliver

The model is small and its output varies a lot between seeds: one candidate can be a clean, readable item and the next a blob or a different object. So do not hand over files unseen. Open the preview image, judge each candidate the way a player would see it in an inventory slot, and pick the one that reads most clearly as the item.

To compare finalists from different runs in one image:

```bash
python scripts/generate.py --preview-of sprites/a_5_2.png sprites/b_9_1.png --out sprites
```

If you cannot view images in your environment, say so and give the user the preview paths so they can choose.

## Writing prompts

Start with a plain item name: `iron pickaxe`, `铜锭`, `strawberry cake`, `闹钟`. The model was trained on item names, optionally with a color in front, so that is what it follows best. It does not follow sentences with details, counts or positions ("a shield with a star on the left"). Chinese and English both work. Put the main noun last; name completion matches on the end of the prompt.

The model knows two kinds of things, and they behave differently:

1. **Items that are common in mods**: tools, weapons, armor, ingots, gems, potions, food. Just name them. These are drawn in the hand-drawn mod look (`style` 0), and `name_completed` is usually `false`, which is normal.
2. **Everyday objects** from a list of about 4,000 (alarm clock, submarine, headphones, umbrella…). The script expands the name to the exact form used in training and switches to the look those were learned in (`style` 1); `sent_to_model` shows the result and `name_completed` is `true`. These tend to fill the whole canvas and look a little softer than mod-style items, so a set that mixes both kinds will not look perfectly uniform. Mention that to the user rather than forcing one style on everything: everyday objects were only learned in style 1 and fall apart at `--style 0`.

**Colors and common materials work.** A color in front of the name is followed well: `蓝色的雨伞`, `purple sword`. So are the usual material words on tools and weapons: `iron pickaxe`, `golden axe`, `铜扳手`. An unusual combination can still let the material take over the whole picture, so that you get a rod or an ingot of that material instead of the item. When that happens, first generate the bare noun (`扳手`) to confirm the model can draw the shape, then ask for the material's color instead: `铜色的扳手`, `橙色的扳手`.

`--search` lists the names the model knows that contain some text, with the full name each maps to:

```bash
python scripts/generate.py --search camera
```

Each result shows `training_sprites`, so you can prefer a name the model knows well. Use it for everyday objects that come out wrong, to find the specific name the model was trained on (`film camera` rather than `camera thing`). For tool-like items it can backfire: a listed name such as `活动扳手` moves the prompt to the everyday-object look and drops any color or material you added, and the bare noun in mod style may draw better. Compare both before settling.

## Working from a reference picture

When the user has a picture of the thing — a photo, a product shot, concept art, a larger drawing — pass it with `--ref` and the sprites follow its shape and colors instead of the model's own idea of the item:

```bash
python scripts/generate.py "旱冰鞋" --ref roller_skate.jpg --out sprites --count 4
```

The script cuts the object out, shrinks it to a 16×16 draft, and lets the model redraw that draft as a clean sprite. The picture should show one object. How the background was removed is reported as `reference_background`:

- `transparent` or `plain`: the picture already had a transparent or single-color background, which is removed exactly.
- `model`: the picture is a photo with a real background, and a small salient-object model (`matting.onnx` in the model folder) picked out the main object. This works when one object clearly stands out. With several objects in view, or one that fills the whole frame, it can take the wrong thing or fail with an `error`; a crop around the object usually fixes that.

If the model folder has no `matting.onnx`, a photo is rejected with an `error` that says so. Then ask the user for the file, a cut-out, or a plain-background picture rather than retrying.

- Still give the item's name as the prompt. It tells the model what it is cleaning the draft up into, and a wrong name pulls the result towards the wrong object.
- The preview's first tile is the draft, and the JSON says so in `preview_note`. Look at it first, above all after `model`: if the draft is not the object the user meant, no setting will rescue the candidates. Then judge the candidates against it: a good one keeps the draft's silhouette and colors but has crisp shading and a darker rim. The draft itself is saved as `..._ref.png`; it is a blurry shrink, not a deliverable.
- `--ref-strength` (default 0.65) is how far the model may move from the draft. Lower it to 0.5 when candidates lose the reference's shape; raise it to 0.8 when they look blurry or washed out and the item is one the model knows well. At 1 the reference is ignored.
- `--ref-colors` (default 0.7) pulls the result's hues back to the reference. Use 1 when the exact colors matter, 0 to let the model choose them.

A reference is also the strongest fix for an item the model does not know or keeps drawing as something else. What a reference cannot do is carry detail that does not fit in 16×16: logos, text, thin parts and small patterns disappear. A white object on a white background can lose parts of itself with the background, and a pale object stays pale.

## When the model does not know the item

This is where you add the most: the model only matches names by their spelling, while you know what things look like. Signs that the name is not known: an everyday object with `name_completed: false`, a low `training_sprites`, or candidates that are plainly some other object (asking for `保温杯` and getting mugs and plant pots).

Rewrite the request into something the model does know, in this order:

1. **A known name for the same thing.** Think of synonyms and the other language, and check them with `--search`: `保温杯` is not listed, but `保温瓶` (vacuum flask) is.
2. **A well-known look-alike plus a color.** `扭蛋` (capsule toy) is unknown and comes out as eggs and specks; `红色的复活节彩蛋` (red easter egg, a listed name with a color in front) gives colored eggs that can stand in for one. Check the look-alike on its own first: a listed name is not always drawn well (`胶囊`, capsule, comes out as little flasks).
3. **A short visual description**: colors, then one plain, concrete noun for the overall shape, placed last. `银色的圆柱形金属杯子` (silver cylindrical metal cup) draws plain cups and beakers that can stand in for a thermos cup. Keep it to a handful of words, and expect the shape noun to be followed more closely than a two-color pattern.

What does not work in a description, and only adds noise: abstract category words (`厨房电器`, "kitchen appliance"), parts and details ("with four indicator lights", "with a toy inside"), counts and positions. Describe the silhouette a player would see in a 16×16 slot, nothing finer.

When you deliver a sprite made this way, say what you actually asked the model for ("drawn as a black battery"), so the user knows it is a stand-in and can object.

## When results stay poor

Work through these, looking at the previews each time:

1. Another run or two with new seeds and a larger `--count`.
2. A different phrasing: the bare noun, a color instead of a material, the other language, a name from `--search`, or a look-alike or short visual description as described above.
3. `--guidance 4` for more varied candidates. Raising guidance rarely rescues a prompt; it makes the model lean harder on whichever word is already dominating.
4. `--style 0` or `--style 1` when the automatic choice looks wrong for a mod-style item.
5. The other network: `--network mo` when Jang's candidates are broken or all the same, `--network jang` when Mo draws a different object.

Give a phrasing two or three runs, and stop after about three phrasings. Some things are beyond the model: roughly three quarters of the 4,000 or so listed everyday objects come out reliably, the rest often come out as "something of that kind", and thin or intricate objects are hard to show in 16×16 at all. When you stop, tell the user plainly that this is the closest of many, and show the two or three best candidates, including ones that got the shape right but the color wrong or the reverse. Recoloring or touching up a sprite by hand is their call, not something to do silently.

## Delivering

- Copy the chosen 16×16 PNG to where the user needs it. In a mod that is usually `src/main/resources/assets/<modid>/textures/item/<item_id>.png`. Name the file after the item id, which is lowercase ASCII with underscores (`copper_wrench.png`); for a Chinese prompt you choose that id yourself or ask.
- Keep the file as it is. Resizing or re-encoding with smoothing ruins pixel art; if a larger image is wanted for display, scale it by a whole number with nearest-neighbor.
- Report the prompt and seed of each delivered sprite, so the user can reproduce or vary it.

## License note

The model weights are released under CC BY-NC-SA 4.0: non-commercial use, with attribution, share-alike. If the user's project is commercial, tell them before they build on the generated sprites. The script itself is MIT-licensed.
