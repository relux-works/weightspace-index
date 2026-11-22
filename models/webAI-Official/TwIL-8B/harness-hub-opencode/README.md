# Harbor OpenCode LoRA

RL-trained rank-32 LoRA adapter for `webAI-Official/TwIL-8B` behind the **OpenCode** harness on data-analysis tasks (TRL asynchronous GRPO). Non-commercial license inherited from the base model.

## Notes

- 32% fewer tool calls on solved SmolDataEnvs tasks; traj2envs repository tasks remain within noise (-5.7 +/- 17.9 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/TwIL-8B-opencode](https://huggingface.co/harness-hub/TwIL-8B-opencode)
- GGUF LoRA: [harness-hub/TwIL-8B-opencode-GGUF](https://huggingface.co/harness-hub/TwIL-8B-opencode-GGUF)
