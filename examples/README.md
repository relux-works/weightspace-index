# Synthetic example tree

**Nothing here is a real model, patch or result.** Every repository, digest, score and author is made up, except `replacements.safetensors`, which is a real 344-byte file so the tests can check a hosted file end to end.

`models/` has the same shape as the repository's own `models/` tree (see [structure](../docs/structure.md#target-shape)). Generated files are built from the root `models/` only, so nothing in this folder ever appears in `bases.json`, `by-digest/`, `by-repo/` or a client listing.

| Path under `models/example-org/Example-Model-1B/` | Shows |
|---|---|
| `base.json` | Two profiles: the unquantized `bf16` upload and a GGUF quant from another publisher |
| `runs/baseline-bf16/` | A baseline run, measured once and reused by every patch |
| `better-math/` | A `replacement` patch: one sparse tensor hosted in this repository, plus a derived GGUF LoRA on Hugging Face, a run, a comparison and an UNKNOWN assessment |
| `shorter-answers/` | A `lora` patch on Hugging Face with two versions, the first deprecated |

To start a real patch, copy the folder shape, not the values: every digest must come from the intake tool.
