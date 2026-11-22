# Harbor Hermes Agent LoRA

RL-trained rank-32 LoRA adapter for `LiquidAI/LFM2.5-24B-A2B` behind the **Hermes Agent** harness on data-analysis tasks (TRL asynchronous GRPO).

## Notes

- 13% fewer tool calls on solved SmolDataEnvs tasks; lower on harder traj2envs repository tasks (-7.7 +/- 8.5 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/LFM2.5-24B-A2B-hermes-v2](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-hermes-v2)
- GGUF LoRA: [harness-hub/LFM2.5-24B-A2B-hermes-v2-GGUF](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-hermes-v2-GGUF)
