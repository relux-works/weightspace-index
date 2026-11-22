# Harbor OpenCode LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM3-1.7B` behind the **OpenCode** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- Strong gain on data-analysis tasks (+15.3 pp, -43% tool calls); on 40 harder repository tasks only 1 -> 2 tasks solved (2.5% -> 5.0%).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM3-1.7B-opencode](https://huggingface.co/harness-hub/LFM3-1.7B-opencode)
- GGUF LoRA: [harness-hub/LFM3-1.7B-opencode-GGUF](https://huggingface.co/harness-hub/LFM3-1.7B-opencode-GGUF)
