# Harbor OpenCode coding agent LoRA

RL-trained rank-32 LoRA adapter for `Qwen/Qwen3.8-27B` built for the [OpenCode](https://opencode.ai) coding agent on software-engineering and terminal tasks.

## Highlights

- With thinking **off**, lifts SWE-bench Pro (hard 51) from 35.3% to 51.0% (+15.7 pp) and Terminal-Bench 2.1 from 46.4% to 52.4% (+6.0 pp).
- Cuts turns to a quarter on SWE-bench Pro (13 vs 48) and half on Terminal-Bench 2.1 on tasks solved by both.
- Ships both PEFT and upstream F16 GGUF LoRA weights.

## Upstream

- PEFT: [harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted)
- GGUF LoRA: [harness-hub/Qwen3.8-27B-opencode-adapted-GGUF](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted-GGUF)
