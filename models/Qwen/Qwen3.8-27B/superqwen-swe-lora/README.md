# SuperQwen SWE-agent LoRA

Rank-16 PEFT LoRA adapter for `Qwen/Qwen3.8-27B`, fine-tuned on 1,200 resolved coding-agent trajectories from `SWE-bench/SWE-smith-trajectories` (`bash`, `str_replace_editor`, `submit`), keeping `<think>` reasoning active on every turn.

## Caveat

The authors' published evaluation checks single next-action tool selection on 40 held-out trajectories (temperature 0 through vLLM); end-to-end SWE-bench task solve rate remains to be measured independently.

## Upstream

- PEFT: [superqwen/qwen3.8-27b-swe-lora](https://huggingface.co/superqwen/qwen3.8-27b-swe-lora)
