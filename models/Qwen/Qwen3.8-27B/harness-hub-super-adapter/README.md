# Harbor Super-Adapter (pi + Hermes + OpenCode)

One rank-96 LoRA adapter for `Qwen/Qwen3.8-27B` formed by a training-free merge of the three specialist RL adapters (`pi`, `Hermes Agent`, `OpenCode`).

## Highlights

- Intended for **thinking on** (`xhigh`): improves mean pass@1 from 55.5% to 63.5% across all six harness–benchmark pairs while cutting tool calls by 19–48%.
- For a single harness with thinking **off**, prefer its dedicated specialist adapter (`harness-hub-pi`, `harness-hub-hermes`, or `harness-hub-opencode`).

## Upstream

- PEFT: [harness-hub/Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter)
- GGUF LoRA: [harness-hub/Qwen3.8-27B-super-adapter-GGUF](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter-GGUF)
