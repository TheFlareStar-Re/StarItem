# sic18_desc30 @ 150,000

This is the description-conditioned training baseline used for subsequent optimization work. The existing SIC-1.2 client package uses `sic16` + `sic17`; its bundled-model manifest does not describe this checkpoint.

## Included files

- `train/train.py`: the original baseline trainer, including description sampling and reduced-memory text encoding.
- `train/baselines/sic18_desc30/config.json`: the recorded training configuration. Its absolute data path is the original training machine's path.
- `train/baselines/sic18_desc30/tokens.json`: the exact token-ID mapping required by this checkpoint.
- `train/baselines/sic18_desc30/manifest.json`: SHA-256 hashes for the checkpoint, configuration, token mapping, training sources and input metadata.
- `tools/check_captions.py`: converts raw bilingual captions to the training input format.
- `train/eval_desc.py`: evaluates held-out descriptions using the existing evaluation pipeline.

The trainer and the core model files match the original `sic18_desc30` archive. Later experiment controls, diagnostic scripts, cloud automation, logs and generated review material are not part of this baseline update. Model weights and training datasets are not included in Git.

## Training recipe

| Setting | Value |
|---|---|
| Training steps | 150,000 |
| Images | 95,913 at 16×16 |
| Text encoder | `intfloat/multilingual-e5-small`, frozen |
| Languages / maximum tokens | Chinese and English / 32 |
| UNet base channels | 128 |
| Trainable token vectors | Enabled; 10,051 entries |
| Description probability | 0.3, when an eligible description exists |
| Description holdout | 3%, selected by image SHA-1 |
| Synthetic name shortening | Disabled |
| Batch size / learning rate | 256 / 0.0002 |
| Warmup / EMA decay | 1,000 / 0.9999 |
| Seed | 0 |

Install PyTorch for the available CUDA driver, then the packages in `train/requirements.txt`. Prepare the same `dataset/clean`, `dataset/synth2`, `dataset/many` and `dataset/manual` directories, each with `metadata.jsonl` and its referenced images. Training runs from the repository root:

```bash
python -X utf8 train/train.py --name sic18_desc30 --ch 128 --extra dataset/synth2 dataset/many dataset/manual --captions dataset/captions.jsonl --cap-desc 0.3 --desc-holdout 3 --tok-emb --cap-aug-synth 0 --seed 0 --steps 150000 --save-every 10000 --max-hours 10
```

`--max-hours 10` is the original run's time limit, not a guarantee that every GPU reaches 150,000 steps within ten hours. If it stops early, repeat the same command with `--resume` until the target step is reached.

`dataset/captions.jsonl` contains one JSON object per line with `dataset`, `file`, `zh`, `en` and `ok` fields. Only records with `ok: true` are used. `dataset` is the dataset directory name and `file` matches its metadata entry. To convert existing raw captions containing `caption_zh` and `caption_en`:

```bash
python -X utf8 tools/check_captions.py path/to/raw_captions.jsonl
```

Compare the input hashes with the manifest before claiming a reproduction of this baseline. Regenerating captions or changing the data changes the experiment. The recorded hashes identify the original files; they are not download links. Training on different hardware or software is not guaranteed to produce byte-identical weights.

## Use the original checkpoint

Place the original `ema_150000.pt` alongside copies of the included `config.json` and `tokens.json` in `runs/sic18_desc30/`. Preserve the token mapping exactly.

Checkpoint SHA-256:

```text
782de29b1951461fac37cf578b78c4bba19ce4e5c3da108f0ef088ffc5d0d6e0
```

Generate a sprite grid:

```bash
python -X utf8 train/sample.py "ruby pickaxe" --run sic18_desc30 --ckpt runs/sic18_desc30/ema_150000.pt --steps 30 --guidance 5.5
```

Export a separate single-network ONNX package, with `onnx` and `onnxruntime` installed:

```bash
python -X utf8 train/export_onnx.py --run sic18_desc30 --ckpt runs/sic18_desc30/ema_150000.pt --name sic18_desc30-150k
```

The existing client sources can consume the exported package; the tracked SIC-1.2 model metadata has not been relabeled as this baseline. This update does not publish a new executable or model download.

## Description evaluation

With the original datasets and checkpoint available, first populate the CLIP feature cache:

```bash
python -X utf8 train/eval_bind.py --run sic18_desc30 --step 150000
```

Then evaluate the held-out descriptions:

```bash
python -X utf8 train/eval_desc.py --run sic18_desc30 --step 150000 --holdout 3
```

The description holdout excludes descriptions from training, not their images or names. Its metric compares a generated image with siblings of the same item; it does not establish generalization to unseen items.

The existing [model license](../LICENSE-MODEL.md) and [data-source attribution](../DATA_SOURCES.md) apply.
