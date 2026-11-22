# Harbor pi LoRA

RL-trained rank-32 LoRA adapter for `webAI-Official/TwIL-8B` behind the **pi** harness on data-analysis tasks (TRL asynchronous GRPO). Non-commercial license inherited from the base model.

## Notes

- Solves tasks about as often as base (+3.4 pp) with 37% fewer tool calls on solved tasks.
- Ships both native PEFT weights and an upstream F16 GGUF LoRA for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/TwIL-8B-pi](https://huggingface.co/harness-hub/TwIL-8B-pi)
- GGUF LoRA: [harness-hub/TwIL-8B-pi-GGUF](https://huggingface.co/harness-hub/TwIL-8B-pi-GGUF)
