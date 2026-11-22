# Harbor Hermes Agent LoRA

RL-trained rank-32 LoRA adapter for `webAI-Official/TwIL-8B` behind the **Hermes Agent** harness on data-analysis tasks (TRL asynchronous GRPO). Non-commercial license inherited from the base model.

## Notes

- 16% fewer tool calls on solved SmolDataEnvs tasks; lower on harder traj2envs repository tasks (-12.9 +/- 17.6 pp).
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/TwIL-8B-hermes](https://huggingface.co/harness-hub/TwIL-8B-hermes)
- GGUF LoRA: [harness-hub/TwIL-8B-hermes-GGUF](https://huggingface.co/harness-hub/TwIL-8B-hermes-GGUF)
