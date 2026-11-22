# Harbor pi LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM3-1.7B` behind the **pi** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- Strong gain on data-analysis tasks (+10.7 pp, -37% tool calls); does not transfer to harder repository tasks (traj2envs -2.5 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM3-1.7B-pi](https://huggingface.co/harness-hub/LFM3-1.7B-pi)
- GGUF LoRA: [harness-hub/LFM3-1.7B-pi-GGUF](https://huggingface.co/harness-hub/LFM3-1.7B-pi-GGUF)
