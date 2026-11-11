# Applying a patch with common runtimes

**These commands were not run.** They show where the runtimes' own documented options fit once every file has been downloaded and checked against `version.json` and `base.json`. Paths are placeholders. The runtimes' documentation, linked below, is the authority; it was read on 10 November 2026, and the vLLM pages track the latest release.

The original model is never modified. Several LoRAs combine the way the runtime combines them; the index adds no rule of its own. A replacement goes into a copy, at most one replacement per copy, and LoRAs can go on top of that copy.

## Replacement: bake a copy

Copy the target profile's safetensors files, confirm each copied file's SHA-256 matches `base.json`, and assign the patch values into the copy: a full tensor replaces the tensor of the same name; a sparse trio first checks that the copy holds the `#old` values at the `#idx` positions, then writes `#new` there. The future client does this step; there is no runtime flag for it. The result is an ordinary model:

```sh
vllm serve ./patched-copy
```

On any other profile, such as a GGUF quant, use the patch's derived LoRA instead.

## LoRA with vLLM

[vLLM's LoRA documentation](https://docs.vllm.ai/en/latest/features/lora/) loads PEFT adapters with `--enable-lora` and names them with `--lora-modules`. A request selects one adapter by name in its `model` field.

```sh
vllm serve ./base-model --enable-lora \
  --lora-modules shorter-answers=./shorter-answers-v2
```

## LoRA with llama.cpp

The [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/26d52b0f9543f441fc2bfa383557c992a4fbf4b9/tools/server/README.md) takes GGUF LoRAs with `--lora`, or `--lora-scaled` with an explicit scale; several adapters add up, so their order does not matter.

```sh
llama-server -m ./base-q4_k_m.gguf --lora ./better-math-lora.gguf
llama-server -m ./base-q4_k_m.gguf \
  --lora-scaled ./better-math-lora.gguf 1.0 --lora-scaled ./shorter-answers.gguf 0.5
```

To merge instead, [`llama-export-lora`](https://github.com/ggml-org/llama.cpp/blob/26d52b0f9543f441fc2bfa383557c992a4fbf4b9/tools/export-lora/README.md) writes a new GGUF:

```sh
llama-export-lora -m ./base-q4_k_m.gguf --lora ./better-math-lora.gguf -o ./patched-q4_k_m.gguf
```

A PEFT LoRA needs a GGUF conversion first; the index lists one as a derived file when it has built it with llama.cpp's converter.

## LoRA with Ollama

An [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) attaches an adapter with `ADAPTER`, on top of the base it names with `FROM`:

```text
FROM ./base-q4_k_m.gguf
ADAPTER ./better-math-lora.gguf
```

```sh
ollama create example-better-math -f Modelfile
```

## Before trusting a result

Loading is not evidence of improvement. Index results apply only to the profile and version they were measured on; a combination of patches has no results unless a run for that exact combination exists.
