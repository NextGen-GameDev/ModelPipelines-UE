# Third-party notices

Model Pipelines from GitHub is licensed under the Apache License 2.0 (`LICENSE`). From Fab, the Fab license applies to the plugin itself. Either way it uses the components below, under their own licenses.

- **Bundled in the plugin:** the tokenizer library (tokenizers-cpp, Hugging Face tokenizers and the Rust crates it links) and six sample photos.
- **Provided by Unreal Engine, not redistributed by us:** ONNX Runtime.
- **Downloaded by you, not shipped in the plugin:** the four sample models.

The full text of the Apache License 2.0 is at the end of this file. It applies to every component below that lists Apache-2.0.

## tokenizers-cpp

`tokenizers_c.lib` is bundled in the Model Pipelines internal tokenizer module. It is built from the fork https://github.com/P1ayer-1/tokenizers-cpp (tag `v0.1.5`) and links the components in this section.

- Origin: mlc-ai/tokenizers-cpp, https://github.com/mlc-ai/tokenizers-cpp. Used through the fork by P1ayer-1, https://github.com/P1ayer-1/tokenizers-cpp.
- SPDX: Apache-2.0
- License text: [Apache License 2.0](#apache-license-20-full-text), at the end of this file.

The tokenizers-cpp fork is licensed under Apache-2.0 (its `LICENSE` file at tag `v0.1.5`).

The internal tokenizer module (`ModelPipelinesTokenizer`) is our own code and is licensed like the rest of this plugin.

## Hugging Face tokenizers (Rust crate 0.21.4)

- Source: https://github.com/huggingface/tokenizers
- SPDX: Apache-2.0
- License text: [Apache License 2.0](#apache-license-20-full-text), at the end of this file.

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
- License text: [Apache License 2.0](#apache-license-20-full-text), at the end of this file.

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

The plugin does not ship model files. The download script fetches them from Hugging Face. All four are licensed under Apache-2.0 (SPDX `Apache-2.0`). The license text is at the end of this file. Details and advice are in [Docs/ModelLicenses.md](Docs/ModelLicenses.md).

| Slug | Hugging Face repo and source URL | Revision | SPDX |
|---|---|---|---|
| `all-MiniLM-L6-v2` | sentence-transformers/all-MiniLM-L6-v2, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 |
| `distilbert-sst2` | distilbert/distilbert-base-uncased-finetuned-sst-2-english, https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english | `714eb0fa89d2f80546fda750413ed43d93601a13` | Apache-2.0 |
| `resnet-50` | microsoft/resnet-50, https://huggingface.co/microsoft/resnet-50 | `34c2154c194f829b11125337b98c8f5f9965ff19` | Apache-2.0 |
| `rtdetr_r18vd` | PekingU/rtdetr_r18vd, https://huggingface.co/PekingU/rtdetr_r18vd | `ac77a11ff0170a41b771c03264987f8ce2b0d753` | Apache-2.0 |

Attribution: each model is by the authors named on its Hugging Face model card (the repo owner above). The models were converted to ONNX, which does not change the license.

Revisions are the full Hugging Face commit hashes the fetch script downloads (the script pins their first eight characters).

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

## Apache License 2.0 (full text)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
