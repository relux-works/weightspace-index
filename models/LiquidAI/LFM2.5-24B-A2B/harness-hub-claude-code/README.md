# Harbor Claude Code LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM2.5-24B-A2B` behind the **Claude Code** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- Published by harness-hub as a reference checkpoint and starting point; the model card marks step-100 evaluation as pending (TODO), with earlier steps performing close to base (0.68 pass@1).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM2.5-24B-A2B-claude-code](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-claude-code)
- GGUF LoRA: [harness-hub/LFM2.5-24B-A2B-claude-code-GGUF](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-claude-code-GGUF)
