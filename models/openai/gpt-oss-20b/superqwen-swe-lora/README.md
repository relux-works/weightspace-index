# SuperQwen SWE-agent LoRA (gpt-oss-20b)

Rank-16 PEFT LoRA adapter trained on 1,200 resolved `SWE-bench/SWE-smith-trajectories` rendered in harmony format (`bash`, `str_replace_editor`, `submit`), targeting attention projections and MoE expert weights on three layers.

## Base note & caveats

- Trained on `superqwen/gpt-oss-20b`, whose three `.safetensors` shards are SHA-256 identical to `openai/gpt-oss-20b` at the pinned revision.
- The authors report that this adapter drops `reasoning` text (92.5% → 0%) and goes straight to tool calls; only single-step next-action accuracy on 40 held-out trajectories was evaluated.

## Upstream

- PEFT: [superqwen/gpt-oss-20b-swe-lora](https://huggingface.co/superqwen/gpt-oss-20b-swe-lora)
