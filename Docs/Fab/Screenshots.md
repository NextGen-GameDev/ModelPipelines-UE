# Fab screenshots: shot list

Resolution and format: captured at 1920x1080, 16:9, PNG. Check Fab's size, file type and count limits in the publisher portal at upload.

General framing: hide the editor panels you do not need, use a clean dark theme, no personal paths in the Output Log, and no other plugins visible.

## Model Pipelines (8 shots)

| # | Show | Where | Framing | Caption |
|---|---|---|---|---|
| 1 | Semantic search with results | Play `L_SemanticSearch` with `all-MiniLM-L6-v2` imported | Game view, a query typed and the ranked results visible | Search by meaning: sentences ranked by similarity to your query. |
| 2 | Sentiment | Play `L_Sentiment` | Game view with a few sentences, labels and scores (for example POSITIVE 0.99) | Classify text and get a label and a score. |
| 3 | Scene classifier | Play `L_SceneClassifier` | Game view with a sample photo and its top labels | Label a picture with the top-K classes. |
| 4 | Detection with boxes | Play `L_Detection` with `detect_dog_walk` or `detect_street_market` | Game view, boxes and labels drawn on the photo | Object detection: boxes, labels and scores. |
| 5 | The import menu | Content Browser, right-click, **Import Hugging Face Model Folder...** | Menu open over a folder with a model, cursor on the entry | Import a Hugging Face model folder in one click. |
| 6 | Package details panel | Open a model package asset (for example `/Game/Models/resnet-50`) | The Details panel with the **License** field filled (SPDX id, source URL, attribution) | Each package records its license. |
| 7 | Blueprint graph with async nodes | Blueprint in `L_SemanticSearch` (or a scratch graph) with **Embed Texts** and the result pins | Graph editor, zoomed so pin names are readable | Call a model with one Blueprint node. |
| 8 | MCP tools in Claude Code | Claude Code with the Unreal MCP server, UE 5.8, asking to list or benchmark models | Terminal window showing the tool call and the result | Optional: let an AI agent import, validate and benchmark models. UE 5.8 only. |

## Tokenizers (6 shots)

| # | Show | Where | Framing | Caption |
|---|---|---|---|---|
| 1 | Blueprint graph: load, encode, decode | A Blueprint that creates a Tokenizer Wrapper (small helper function), then **Initialize Tokenizer From File**, **Encode**, **Decode** | Whole graph, readable pin names | Load a tokenizer.json and encode text in four nodes. |
| 2 | Encode result | Same Blueprint, **Print String** of the ids after a play-in-editor run | Viewport with printed ids next to the graph | Token ids from a Hugging Face tokenizer.json. |
| 3 | Padded batch | C++ snippet or Blueprint calling `EncodeBatch` | Code or graph showing `InputIds` and `AttentionMask` | Padded, truncated batches with an attention mask. |
| 4 | Error handling | Blueprint printing **Get Last Error** after loading bad JSON | Output Log and graph | Bad input returns an error message, not a crash. |
| 5 | Plugins window | **Edit > Plugins**, search "Tokenizers", category **AI** | The plugin row with the Tokenizers entry | Enable it in the Plugins window. |

## Capture status (2026-10-07, WP8.3 re-capture)

Captured PNGs and the manual checklist are in the private folder `.pipelines-dev/fab-shots/` (not in the repo). WP8.1 versions are in its `v1/` folder.

| Shot | Status |
|---|---|
| Model Pipelines 1, 2 | captured (PIE, scripted; camera pitched so the floor is about a quarter of the frame) |
| Model Pipelines 3 (scene classifier) | captured (soccer ball board, top label soccer ball 100.0%) |
| Model Pipelines 4 (detection) | captured (dog-walk board centred) |
| Model Pipelines 5 (import menu) | manual (the editor window did not react to scripted or real clicks again) |
| Model Pipelines 6 (package details) | captured |
| Model Pipelines 7 (Blueprint graph) | captured (demo graph, not the sample's own; Package comes from a variable) |
| Model Pipelines 8 | skipped (Noah's) |
| Tokenizers 1, 3, 4, 5 | captured (1 and 3 and 4 create the object in a small `Create Tokenizer` function, so the caption's "Construct Object from Class" node is not on screen; 3 breaks the batch to show `Input Ids` and `Attention Mask`; 4 shows the graph plus the Output Log lines from a play-in-editor run) |
| Tokenizers 2 (encode result) | manual |
| Tokenizers 6 | dropped (release zips contain no tests) |
