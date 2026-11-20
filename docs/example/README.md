# Synthetic example

Open this synthetic base to see what a filled-in patch folder looks like. This documentation example also supplies test fixtures.

**Nothing here is a real model, patch or result.** Every repository, digest, score and author is made up, except `replacements.safetensors`, which is a real 344-byte file so the tests can check a hosted file end to end.

`models/` has the same shape as the repository's own `models/` tree (see [structure](../structure.md#target-shape)). It is a tree of its own: `models/index.json`, `models/README.md`, `lookup/by-digest/`, `lookup/by-repo/` and the base's `index.json` and `README.md` here are generated from this folder's `models/`, exactly as the root ones are from the root `models/`. The root's generated files never read this folder, so nothing here appears in a client listing.

| Path under `models/example-org/Example-Model-1B/` | Shows |
|---|---|
| `base.json` | Two profiles: the unquantized `bf16` upload and a GGUF quant from another publisher |
| `runs/baseline-bf16/` | A baseline run, measured once and reused by every patch |
| `better-math/` | A `replacement` patch: one sparse tensor hosted in this repository, plus a derived GGUF LoRA on Hugging Face, a run, a comparison and an attempted-but-not-established UNKNOWN assessment |
| `shorter-answers/` | A `lora` patch on Hugging Face with two versions, the first deprecated |
| `polite-tone/`, `step-by-step/` | Two more `lora` patches on Hugging Face, so the model page has stacks to compare |

To start a real patch, copy the folder shape, not the values: every digest must come from the intake tool. Real patch slugs follow the [author-prefixed sourced-naming rule](../structure.md#rules); the synthetic names here are not naming sources for real submissions.

`bench/combinations/example-org/Example-Model-1B/` holds five made-up [combination studies](../benchmarking.md#combination-studies): patches that help each other, add up, get in each other's way, a report too small to tell, and a three-patch stack filed without solo scores.
