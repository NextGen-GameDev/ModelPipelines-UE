# Third-party notices

Model Pipelines is licensed under the Apache License 2.0 (`LICENSE` in this folder). It uses the components below.

- **Bundled in the plugin:** the tokenizer library (tokenizers-cpp, Hugging Face tokenizers and the Rust crates it links) and six sample photos.
- **Provided by Unreal Engine, not redistributed by us:** ONNX Runtime.
- **Downloaded by you, not shipped in the plugin:** the four sample models.

The full text of the Apache License 2.0 is in `LICENSE` in this folder. It applies to every component below that lists Apache-2.0.

## tokenizers-cpp

`tokenizers_c.lib` is bundled in the Model Pipelines internal tokenizer module. It is built from the fork https://github.com/P1ayer-1/tokenizers-cpp (tag `v0.1.5`) and links the components in this section.

- Origin: mlc-ai/tokenizers-cpp, https://github.com/mlc-ai/tokenizers-cpp. Used through the fork by P1ayer-1, https://github.com/P1ayer-1/tokenizers-cpp.
- SPDX: Apache-2.0
- License text: `LICENSE`.

The tokenizers-cpp fork is licensed under Apache-2.0 (its `LICENSE` file at tag `v0.1.5`).

The internal tokenizer module (`ModelPipelinesTokenizer`) is NextGen-GameDev code under Apache-2.0, like the rest of this plugin.

## Hugging Face tokenizers (Rust crate 0.21.4)

- Source: https://github.com/huggingface/tokenizers
- SPDX: Apache-2.0
- License text: `LICENSE`.

## Oniguruma (bundled C code in the `onig_sys` 69.9.3 crate)

- License: BSD-2-Clause
- Source: https://github.com/kkos/oniguruma (vendored by https://github.com/rust-onig/rust-onig)

Verbatim from `oniguruma/COPYING` in onig_sys 69.9.3:

```
Oniguruma LICENSE
-----------------

Copyright (c) 2002-2021  K.Kosako  <kkosako0@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

## esaxx-rs 0.1.10 (bundles C++ esaxx)

- SPDX: Apache-2.0
- License text: `LICENSE`.

## Other Rust crates

The other 72 Rust crates linked into `tokenizers_c.lib` are under permissive licenses. By count:

| License (SPDX expression as listed by the crate) | Crates |
|---|---|
| MIT OR Apache-2.0 | 49 |
| MIT | 10 |
| MIT/Apache-2.0 | 7 |
| Apache-2.0 | 3 |
| Unlicense OR MIT | 2 |
| Apache-2.0 OR MIT OR Zlib | 2 |
| Apache-2.0 OR BSL-1.0 | 1 |
| (MIT OR Apache-2.0) AND Unicode-3.0 | 1 |
| BSD-2-Clause OR Apache-2.0 OR MIT | 1 |

Regenerate the full list with `cargo metadata --locked` in the fork's `rust/` folder.

The crate tree is unchanged from tag `v0.1.4` to `v0.1.5` (`Cargo.lock` and `Cargo.toml` are identical), so these counts hold for the lib ModelPipelines bundles.

## ONNX Runtime

ONNX Runtime is provided by Unreal Engine's NNERuntimeORT plugin. Model Pipelines does not redistribute it. The version depends on the engine:

| Engine | ONNX Runtime | Source (from the engine's `.tps` file) |
|---|---|---|
| UE 5.8 | 1.24.1 (DirectML build) | https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.24.1 |
| UE 5.7 | 1.20 (the engine file names 1.20.0, DirectML build) | https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.20.0 |

- License: MIT (as named by the package's license page, https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.1/License). The engine's `.tps` files give the license by link only.
- Copyright: Microsoft Corporation.

Courtesy copy of the MIT license:

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Checked 2026-10-06 against the license of Microsoft.ML.OnnxRuntime.DirectML 1.24.1 on nuget.org (MIT, "Copyright (c) Microsoft Corporation").

## Sample models

The plugin does not ship model files. The download script fetches them from Hugging Face. All four are licensed under Apache-2.0 (SPDX `Apache-2.0`). The license text is in `LICENSE`. Details and advice are in [Docs/ModelLicenses.md](Docs/ModelLicenses.md).

| Slug | Hugging Face repo and source URL | Revision | SPDX |
|---|---|---|---|
| `all-MiniLM-L6-v2` | sentence-transformers/all-MiniLM-L6-v2, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 |
| `distilbert-sst2` | distilbert/distilbert-base-uncased-finetuned-sst-2-english, https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english | `714eb0fa89d2f80546fda750413ed43d93601a13` | Apache-2.0 |
| `resnet-50` | microsoft/resnet-50, https://huggingface.co/microsoft/resnet-50 | `34c2154c194f829b11125337b98c8f5f9965ff19` | Apache-2.0 |
| `rtdetr_r18vd` | PekingU/rtdetr_r18vd, https://huggingface.co/PekingU/rtdetr_r18vd | `ac77a11ff0170a41b771c03264987f8ce2b0d753` | Apache-2.0 |

Attribution: each model is by the authors named on its Hugging Face model card (the repo owner above). The models were converted to ONNX, which does not change the license.

Revisions are the full Hugging Face commit hashes the fetch script downloads (the script pins their first eight characters).

## Phonon-2 (optional user-downloaded model)

The plugin does **not** ship Phonon-2. You download it and convert it with `Tools/` (see [Docs/SpeechToText.md](Docs/SpeechToText.md)). If your game includes the converted files, you must reproduce this attribution:

- Model: Phonon-2, Copyright 2026 Fermion Research. Source: https://huggingface.co/FermionResearch/Phonon-2
- SPDX: CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/
- Derived from NVIDIA Parakeet TDT 0.6B v3 (nvidia/parakeet-tdt-0.6b-v3), (c) NVIDIA Corporation, CC-BY-4.0: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Changes: converted to ONNX (the `fp16w` variant also stores weights as 16-bit numbers). Fermion's own changes to the base model are described in the `NOTICE` file that comes with the model.
- Training data includes CHiME-6 (CC-BY-SA-4.0, https://chimechallenge.github.io/chime6/) and SPGISpeech (Kensho's public terms, https://datasets.kensho.com/datasets/spgispeech).

Keep the model's `NOTICE` and `LICENSE-WEIGHTS-CC-BY-4.0.txt` files with your game.

## Sample photos

The six photos in `Content/Samples/Textures` are CC0 1.0 (SPDX `CC0-1.0`) or public domain, from Wikimedia Commons. CC0 needs no attribution. We list the authors anyway.

| Photo | Author | License | Source |
|---|---|---|---|
| classify_jellyfish | CGiblets | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Pair_of_Jellyfish_Float_Happily.jpg |
| classify_soccer_ball | Rakoon | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Ball_soccer_1990s2.jpg |
| classify_violin | Just plain Bill | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Violin_VL100.png |
| classify_elephant | Nanjil Bala | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Thirukurungkudi_Temple_Elephant.jpg |
| detect_dog_walk | Tomwsulcer | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Fort_Greene_Brooklyn_NY_assorted_photos_near_Fulton_Street_2_lady_walking_four_dogs.jpg |
| detect_street_market | Syced | CC0-1.0 | https://commons.wikimedia.org/wiki/File:Street_market_in_Hanoi,_2003_July_41.jpg |
