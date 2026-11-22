# Harbor Hermes Agent LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM3-1.7B` behind the **Hermes Agent** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- Strong gain on data-analysis tasks (+16.0 pp, -38% tool calls); task-specific (traj2envs -7.5 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM3-1.7B-hermes](https://huggingface.co/harness-hub/LFM3-1.7B-hermes)
- GGUF LoRA: [harness-hub/LFM3-1.7B-hermes-GGUF](https://huggingface.co/harness-hub/LFM3-1.7B-hermes-GGUF)
