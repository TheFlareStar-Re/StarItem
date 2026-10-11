"""下载生成合成训练数据用的模型（约 7.8 GB），放进 Hugging Face 的默认缓存目录。

Run it yourself:
  .venv/Scripts/python.exe -X utf8 synth/download_models.py
If huggingface.co is slow or unreachable, set a mirror first (PowerShell):
  $env:HF_ENDPOINT = "https://hf-mirror.com"
Interrupted downloads resume when you run it again.
"""
from huggingface_hub import snapshot_download

# (仓库, 只取这些文件)。SDXL 只要半精度的权重，不下全精度和其他框架的版本
MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]),
    ("madebyollin/sdxl-vae-fp16-fix", ["config.json", "diffusion_pytorch_model.safetensors"]),
    ("nerijs/pixel-art-xl", ["pixel-art-xl.safetensors"]),
    ("latent-consistency/lcm-lora-sdxl", ["pytorch_lora_weights.safetensors"]),
]


def main():
    import sys

    # --base-only：只下 SDXL 和 VAE（用自己的 LoRA 出图时够用了）
    models = MODELS[:2] if "--base-only" in sys.argv else MODELS
    total = 0
    for repo, patterns in models:
        print(f"== {repo}", flush=True)
        path = snapshot_download(repo, allow_patterns=patterns)
        size = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*") if f.is_file())
        total += size
        print(f"   {size / 1e9:.2f} GB -> {path}", flush=True)
    print(f"done, {total / 1e9:.2f} GB in total")


if __name__ == "__main__":
    main()
