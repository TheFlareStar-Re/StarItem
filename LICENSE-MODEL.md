# Model license

The SIC model weights (`unet.onnx`, `unet_names.onnx`, and the model files bundled inside the StarItem executable) are licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

Copyright (c) 2026 TheFlareStar

- License summary: https://creativecommons.org/licenses/by-nc-sa/4.0/
- Full legal text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

In short: you may share and adapt the weights for non-commercial purposes, as long as you give credit and
distribute your contributions under the same license. This summary is not a substitute for the license itself.

## Why this license

The training data includes hand-drawn sprites released under CC BY-NC and CC BY-NC-SA licenses, so the weights
are released for non-commercial use under a share-alike license. Sources and attribution are listed in
[DATA_SOURCES.md](DATA_SOURCES.md).

## What is not covered

- The source code in this repository is licensed separately under the MIT License, see [LICENSE](LICENSE).
- The text encoder distributed with the model is [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
  (MIT License); `text_encoder.onnx` and `tokenizer.json` are converted from it.
- `matting.onnx`, which cuts the object out of a reference photo, is the [U²-Net-p](https://github.com/xuebinqin/U-2-Net)
  salient-object model (Apache License 2.0), in the ONNX form published with [rembg](https://github.com/danielgatis/rembg).
  The full license text is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt), ships next to the file as
  `matting.LICENSE.txt`, and can be read in the client under "Licenses" in the status bar.
