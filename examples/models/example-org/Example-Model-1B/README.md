# example-org/Example-Model-1B

Generated from the files in this folder by `weightspace generate`; do not edit.

## Profiles

| Profile | Source | Files |
|---|---|---|
| `bf16` | [example-org/Example-Model-1B@aaaaaaa](https://huggingface.co/example-org/Example-Model-1B/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) | 2 |
| `example-quants-q4-k-m` | [example-quants/Example-Model-1B-GGUF@ccccccc](https://huggingface.co/example-quants/Example-Model-1B-GGUF/tree/cccccccccccccccccccccccccccccccccccccccc) | 1 |

## Patches

| Patch | Kind | Latest | Summary |
|---|---|---|---|
| [better-math](better-math/) | replacement | v1 | SYNTHETIC EXAMPLE: two replaced weights. |
| [shorter-answers](shorter-answers/) | lora | v2 | SYNTHETIC EXAMPLE: a published PEFT LoRA. |

## Results

Latest active version of each patch, against the baseline run on the same profile.

| Patch | Profile | Metric | Baseline | Patched | Measured by |
|---|---|---|---|---|---|
| better-math v1 | `bf16` | accuracy | 0.5 fraction (n=200) | 0.5 fraction (n=200) | index |
| shorter-answers v2 | `bf16` | not measured | | | |

## Assessments

| Patch | Assessment | Intended use | Decision |
|---|---|---|---|
| better-math v1 | arithmetic-help | Arithmetic help on the bf16 profile. | UNKNOWN |
