# LiquidAI/LFM2.5-24B-A2B

Generated from the files in this folder by `weightspace generate`; do not edit.

## Profiles

| Profile | Source | Files |
|---|---|---|
| `bf16` | [LiquidAI/LFM2.5-24B-A2B@3dc4cd0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/tree/3dc4cd09c9a2a99e1e819693e79d3aba5e9bb4f2) | 10 |

## Patches

| Patch | Kind | Latest | Summary |
|---|---|---|---|
| [harness-hub-claude-code](harness-hub-claude-code/) | lora | v1 | RL-trained reference LoRA for Claude Code on data-analysis tasks (published by authors as a starting point, step 100). |
| [harness-hub-hermes](harness-hub-hermes/) | lora | v1 | RL-trained LoRA for Hermes Agent on data-analysis tasks: similar solve rate to base with 13% fewer tool calls. |
| [harness-hub-opencode](harness-hub-opencode/) | lora | v1 | RL-trained LoRA for OpenCode on data-analysis tasks: matches base solve rate (+1.3 pp) with 30% fewer tool calls. |
| [harness-hub-pi](harness-hub-pi/) | lora | v1 | RL-trained LoRA for the pi harness on data-analysis tasks: matches base solve rate (+2.7 pp) with 20% fewer tool calls. |

## Results

Latest active version of each patch, against the baseline run on the same profile.

| Patch | Profile | Metric | Baseline | Patched |
|---|---|---|---|---|
| harness-hub-claude-code v1 | `bf16` | not measured | | | |
| harness-hub-hermes v1 | `bf16` | not measured | | | |
| harness-hub-opencode v1 | `bf16` | not measured | | | |
| harness-hub-pi v1 | `bf16` | not measured | | | |
