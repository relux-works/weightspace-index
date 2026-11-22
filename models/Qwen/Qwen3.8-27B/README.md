# Qwen/Qwen3.8-27B

Generated from the files in this folder by `weightspace generate`; do not edit.

## Profiles

| Profile | Source | Files |
|---|---|---|
| `bf16` | [Qwen/Qwen3.8-27B@3756181](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) | 18 |

## Patches

| Patch | Kind | Latest | Summary |
|---|---|---|---|
| [harness-hub-hermes](harness-hub-hermes/) | lora | v1 | RL-trained LoRA for Hermes Agent: +14.1 pp on Terminal-Bench 2.1 and +13.7 pp on SWE-bench Pro with thinking off. |
| [harness-hub-opencode](harness-hub-opencode/) | lora | v1 | RL-trained LoRA for the OpenCode harness: matches base xhigh thinking accuracy with thinking off and 6–8x fewer tokens. |
| [harness-hub-pi](harness-hub-pi/) | lora | v1 | RL-trained LoRA for the pi coding agent: matches base xhigh thinking accuracy with thinking off and ~4.5x fewer tokens. |
| [harness-hub-super-adapter](harness-hub-super-adapter/) | lora | v1 | Training-free rank-96 merge of the pi, Hermes and OpenCode LoRAs: improves thinking-on pass@1 across harnesses with 19–48% fewer tool calls. |
| [superqwen-swe-lora](superqwen-swe-lora/) | lora | v1 | Rank-16 SFT LoRA trained on 1,200 resolved SWE-smith trajectories (`bash`, `str_replace_editor`, `submit`); next-action tool match jumps 65% → 97.5%. |

## Results

Latest active version of each patch, against the baseline run on the same profile.

| Patch | Profile | Metric | Baseline | Patched |
|---|---|---|---|---|
| harness-hub-hermes v1 | `bf16` | not measured | | | |
| harness-hub-opencode v1 | `bf16` | not measured | | | |
| harness-hub-pi v1 | `bf16` | not measured | | | |
| harness-hub-super-adapter v1 | `bf16` | not measured | | | |
| superqwen-swe-lora v1 | `bf16` | not measured | | | |
