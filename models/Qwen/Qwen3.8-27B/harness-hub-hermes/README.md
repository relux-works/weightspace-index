# Harbor Hermes Agent LoRA

RL-trained rank-32 LoRA adapter for `Qwen/Qwen3.8-27B` built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) on software-engineering and terminal tasks.

## Highlights

- With thinking **off**, raises Terminal-Bench 2.1 from 44.7% to 58.8% (+14.1 pp) and SWE-bench Pro (hard 51) from 29.4% to 43.1% (+13.7 pp).
- Cuts turns below 40% of the unpatched base on tasks solved by both (22 vs 59 on SWE-bench Pro, 8 vs 21 on Terminal-Bench 2.1).
- Ships both PEFT and upstream F16 GGUF LoRA weights.

## Upstream

- PEFT: [harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted)
- GGUF LoRA: [harness-hub/Qwen3.8-27B-hermes-adapted-GGUF](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted-GGUF)
