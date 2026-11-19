# Qwen/Qwen3.5-0.8B

Generated from the files in this folder by `weightspace generate`; do not edit.

## Profiles

| Profile | Source | Files |
|---|---|---|
| `bf16` | [Qwen/Qwen3.5-0.8B@2fc0636](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | 1 |

## Patches

| Patch | Kind | Latest | Summary |
|---|---|---|---|
| [rikiyat-thinking-loops](rikiyat-thinking-loops/) | replacement | v1 | Maintainer-produced canonical replacement representation of the published edit by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab to the language-model MLP down-projection tensors in layers 15 and 19 of Qwen3.5-0.8B, built from the pinned base and edited checkpoints without changing tensor dtype or replacement values. |

## Results

Latest active version of each patch, against the baseline run on the same profile.

| Patch | Profile | Metric | Baseline | Patched | Measured by |
|---|---|---|---|---|---|
| rikiyat-thinking-loops v1 | `bf16` | not measured | | | |
