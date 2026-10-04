# Model licenses

Model Pipelines runs models you bring. The models have their own licenses. This page lists the licenses of the four sample models and the sample photos, and gives advice for your own models. This is not legal advice.

## Sample models

All four sample models are licensed under Apache-2.0 (SPDX id `Apache-2.0`).

| Slug | Hugging Face repo | Revision | License |
|---|---|---|---|
| `all-MiniLM-L6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243` | Apache-2.0 |
| `distilbert-sst2` | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | `714eb0fa` | Apache-2.0 |
| `resnet-50` | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | `34c2154c` | Apache-2.0 |
| `rtdetr_r18vd` | [PekingU/rtdetr_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd) | `ac77a11f` | Apache-2.0 |

The download script checks each model card and prints the license before it downloads weights. The importer stores the license in the package's **License** field (SPDX id, source URL, attribution).

Model cards may also name the datasets the model was trained on. Read those terms too.

## Sample photos

The photos on the vision sample boards are CC0 or public domain, from Wikimedia Commons. No attribution is required for CC0; we list the authors anyway.

| Photo | Used for | Author | License | Source |
|---|---|---|---|---|
| classify_jellyfish | Scene classifier | CGiblets | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Pair_of_Jellyfish_Float_Happily.jpg) |
| classify_soccer_ball | Scene classifier | Rakoon | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Ball_soccer_1990s2.jpg) |
| classify_violin | Scene classifier | Just plain Bill | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Violin_VL100.png) |
| classify_elephant | Scene classifier | Nanjil Bala | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Thirukurungkudi_Temple_Elephant.jpg) |
| detect_dog_walk | Detection | Tomwsulcer | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Fort_Greene_Brooklyn_NY_assorted_photos_near_Fulton_Street_2_lady_walking_four_dogs.jpg) |
| detect_street_market | Detection | Syced | CC0-1.0 | [page](https://commons.wikimedia.org/wiki/File:Street_market_in_Hanoi,_2003_July_41.jpg) |

## Choosing models for your game

1. Open the model card on Hugging Face and read the license before you ship the model.
2. Permissive licenses (Apache-2.0, MIT, BSD) are fine. Keep the license text and attribution in your game's notices.
3. Avoid AGPL and GPL models. Ultralytics YOLO is an example.
4. Avoid non-commercial licenses (for example CC-BY-NC, and many research weights) in a commercial game.
5. Check the dataset terms the model card mentions.
6. Record the license on the package: open the package asset and fill **License** with the SPDX id, the source URL and the attribution text.
