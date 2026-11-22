# Harbor OpenCode LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM2.5-24B-A2B` behind the **OpenCode** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- 30% fewer tool calls on solved SmolDataEnvs tasks; traj2envs repository tasks remain within noise (-5.0 +/- 13.9 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM2.5-24B-A2B-opencode](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-opencode)
- GGUF LoRA: [harness-hub/LFM2.5-24B-A2B-opencode-GGUF](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-opencode-GGUF)
