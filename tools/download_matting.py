"""下载抠背景用的小模型 U²-Net-p（约 4.6 MB），放到客户端和模型目录里。

The model is the "u2netp" salient-object network (Qin et al., Apache-2.0) in the ONNX form published with rembg.
It is only used to cut the object out of a reference photo whose background is not plain.

Run it yourself:
  python -X utf8 tools/download_matting.py
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
TARGETS = [ROOT / "client" / "models" / "matting.onnx", ROOT / "export" / "SIC-1.1" / "matting.onnx",
           ROOT / "export" / "SIC-1.2" / "matting.onnx"]
NOTICE = """matting.onnx is the U^2-Net-p salient-object model from
"U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
by Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane
and Martin Jagersand (https://github.com/xuebinqin/U-2-Net), in the ONNX form
published with rembg (https://github.com/danielgatis/rembg). The file is
distributed unmodified.

It is licensed under the Apache License, Version 2.0. The full text follows.

"""


def main():
    first = TARGETS[0]
    if not first.exists():
        print(f"downloading {URL}", flush=True)
        tmp = first.with_name(first.name + ".part")
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "StarItem"}), timeout=120) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out)
        tmp.replace(first)
    data = first.read_bytes()
    print(f"{first} ({len(data) / 1e6:.1f} MB, sha256 {hashlib.sha256(data).hexdigest()})")
    if not 3e6 < len(data) < 8e6:
        sys.exit("the file size is not what u2netp.onnx should be (about 4.6 MB); delete it and try again")

    # 用 onnxruntime 试跑一次，确认拿到的确实是这个模型
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(first), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.run(None, {inp.name: np.zeros((1, 3, 320, 320), dtype=np.float32)})[0]
    print(f"input {inp.name} {inp.shape}, first output {out.shape}")
    if out.shape != (1, 1, 320, 320):
        sys.exit("unexpected output shape; this is not the u2netp model")
    for target in TARGETS[1:]:
        if target.parent.is_dir():
            shutil.copyfile(first, target)
            print(f"copied to {target}")
    # 许可证全文跟着模型文件走
    license_text = NOTICE + (ROOT / "licenses" / "Apache-2.0.txt").read_text(encoding="utf-8")
    for target in TARGETS:
        if target.parent.is_dir():
            target.with_name("matting.LICENSE.txt").write_text(license_text, encoding="utf-8", newline="\n")
    print("done")


if __name__ == "__main__":
    main()
