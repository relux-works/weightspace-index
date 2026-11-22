# Harbor pi LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM2.5-24B-A2B` behind the **pi** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- 20% fewer tool calls on solved SmolDataEnvs tasks; traj2envs repository tasks remain within noise (-5.0 +/- 13.9 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM2.5-24B-A2B-pi-v2](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi-v2)
- GGUF LoRA: [harness-hub/LFM2.5-24B-A2B-pi-v2-GGUF](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi-v2-GGUF)
