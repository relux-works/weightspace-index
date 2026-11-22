# Harbor pi coding agent LoRA

RL-trained rank-32 LoRA adapter for `Qwen/Qwen3.8-27B` built for the [pi](https://github.com/earendil-works/pi) coding agent on software-engineering and terminal tasks.

## Highlights

- Trained with TRL asynchronous GRPO inside real sandboxed `pi` sessions.
- With thinking **off**, reaches 64.7% on SWE-bench Pro (hard 51) and 55.8% on Terminal-Bench 2.1, generating ~4.5× fewer tokens per task than the base model at `xhigh` reasoning effort.
- Ships both native PEFT weights (`adapter_model.safetensors`, 638 MB) and an upstream F16 GGUF LoRA (`Qwen3.8-27B-pi-lora-F16.gguf`, 319 MB) for `llama.cpp`.

## Upstream

- PEFT: [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted)
- GGUF LoRA: [harness-hub/Qwen3.8-27B-pi-adapted-GGUF](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF)
