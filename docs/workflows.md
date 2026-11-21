# Applying a patch with common runtimes

On the patch card, choose a download and select your runtime. The card shows the launch command after setup. **Copy full recipe** includes the download, checksum verification, base preparation and launch steps. Run the recipe in a fresh working directory with the selected runtime installed. With JavaScript disabled, download the file first and expand the recipe under the runtime name. The [Qwen Thinking loops card](../models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/README.md) has one bundle containing PEFT and GGUF adapters for the pinned base. The included provenance records a one-token llama.cpp server completion, a Transformers/PEFT forward pass and a vLLM CPU adapter-loader check. End-to-end vLLM serving remains unverified.

The generic examples below show where the runtimes' own documented options fit once every file has been downloaded and checked against `version.json` and `base.json`. Paths are placeholders. The runtimes' documentation, linked below, is the authority; it was read on 10 November 2026, and the vLLM pages track the latest release.

The original model is never modified. Several LoRAs combine the way the runtime combines them; the index adds no rule of its own. A replacement goes into a copy, at most one replacement per copy, and LoRAs can go on top of that copy.

## Replacement: bake a copy

Copy the target profile's safetensors files, confirm each copied file's SHA-256 matches `base.json`, and assign the patch values into the copy: a full tensor replaces the tensor of the same name; a sparse trio first checks that the copy holds the `#old` values at the `#idx` positions, then writes `#new` there. The current `weightspace apply` command does this step. there is no runtime flag for it. The result is an ordinary model:

```sh
vllm serve ./patched-copy
```

For another profile, such as a GGUF quant, use the patch's LoRA when the runtime supports that adapter and base model.

## Attached and merged LoRAs

The Qwen bundle uses F32 `selector-exact` factors, with rank 64 for each of the two edited tensors. The site's Transformers/PEFT, llama.cpp and vLLM recipes load the factors as separate adapter weights. During prompt processing and generation, each adapted layer computes `B(Ax)` for the token activations and adds it to the base layer's output. The additional weights take up inference memory. [PEFT forward pass](https://github.com/huggingface/peft/blob/v0.21.3/src/peft/tuners/lora/layer.py#L977-L1009), [llama.cpp inference graph](https://github.com/ggml-org/llama.cpp/blob/ece963f41b0b02d7a0d61436ae365762c073a4c8/src/llama-graph.cpp#L1373-L1400), [vLLM linear layers](https://docs.vllm.ai/en/stable/api/vllm/lora/layers/base_linear/).

An explicit merge applies the scaled `B·A` update to the base weights once before inference. PEFT's [`merge_and_unload()`](https://huggingface.co/docs/peft/main/package_reference/lora#merge-lora-weights-into-the-base-model) and llama.cpp's export command below produce weights for ordinary inference. A merge can round the computed values. `weightspace apply` writes the published replacement values bit for bit.

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
