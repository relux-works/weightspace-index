# Workflow examples: verified inputs before a runtime

**WORKFLOW EXAMPLES ONLY — no real patch is supplied or supported. Every example's entry quality remains UNKNOWN until exact-profile assessment evidence exists.** These command shapes were not executed. Paths are illustrative relative staging locations, not available artifacts. They do not install software, download a model or qualify any model architecture, hardware, dtype or patch.

**Documented** below means a cited primary source describes the runtime interface. **Inferred** means an index workflow requirement or conditional composition of those interfaces. Source inspection is not a model test. Primary runtime sources were accessed **10 November 2026 UTC**; the vLLM `latest` pages are mutable, so any later real entry must pin and qualify its installed release/build rather than treating this access date as a build identity.

## Native safetensors with vLLM

**Documented:** vLLM's [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) accept a Hugging Face model name or local path, and `--load-format safetensors` selects safetensors loading. Its [online-serving documentation](https://docs.vllm.ai/en/latest/serving/online_serving/) describes serving and requires an appropriate chat template for chat requests. These facts do not establish support for an arbitrary model directory.

**Inferred workflow:** pin the reviewed index commit; select a native profile; manually obtain its exact HF revision or in-repository files; verify SHA-256/size for all shards, indexes, config, tokenizer and template. Check rights and complete-file coverage. No special index tool is needed. An advertised digest remains advertised in the record until a retained complete-file verification establishes more.

For a **sparse exact replacement**, first require a separately qualified applicator and expected tensor/output checks. Preserve the base and create a complete candidate directory. For a **whole prepared artifact**, verify the whole delivered directory and transformation manifest. Neither kind uses LoRA flags simply because its files are safetensors. Missing patch identity or qualified application route stops the workflow here.

Only after those gates, a separately authorized load check could use:

```sh
# WORKFLOW ONLY; illustrative prepared local directory; quality UNKNOWN.
vllm serve ./staging/candidate-safetensors --load-format safetensors
```

**Documented, LoRA only:** [vLLM's LoRA documentation](https://docs.vllm.ai/en/latest/features/lora/) describes compatible models, `--enable-lora` and named adapter paths through `--lora-modules`. A separate LoRA workflow could have this shape after base, adapter/configuration, targets, scale and runtime qualification:

```sh
# LORA WORKFLOW ONLY; not a sparse-edit application; quality UNKNOWN.
vllm serve ./staging/base-safetensors --load-format safetensors \
  --enable-lora --lora-modules example-adapter=./staging/lora-adapter
```

**Inferred evidence gate:** retain the installed runtime/build, compute/cache precision, effective decoding and template settings for each arm. A successful load would establish loading only. Native quality stays UNKNOWN until matched base/candidate observations support a scoped assessment; no timing, memory-fit or benefit claim is made here.

## GGUF with llama.cpp

**Documented:** the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/26d52b0f9543f441fc2bfa383557c992a4fbf4b9/tools/server/README.md) lists `-m` for a local model and `--lora` for an adapter. The [adapter exporter documentation](https://github.com/ggml-org/llama.cpp/blob/26d52b0f9543f441fc2bfa383557c992a4fbf4b9/tools/export-lora/README.md) documents a separate base-plus-adapter export. Those command interfaces do not qualify an indexed treatment.

**Inferred workflow:** pin the index snapshot and exact GGUF profile, including tensor mixture, conversion/quantization recipe, tokenizer/template and runtime/build. Obtain the named in-repository file or immutable external file manually; verify every required SHA-256 and size. A native safetensors assessment cannot qualify this GGUF, even when its display name is similar.

For a **whole GGUF artifact**, verify the output inventory and provenance manifest. A separately authorized load check could then have this shape:

```sh
# WORKFLOW ONLY; illustrative verified whole GGUF; quality UNKNOWN.
llama-server -m ./staging/candidate.gguf
```

For a **LoRA adapter**, runtime application is a different treatment:

```sh
# LORA WORKFLOW ONLY; compatible base/adapter must be qualified; quality UNKNOWN.
llama-server -m ./staging/base.gguf --lora ./staging/lora-adapter.gguf
```

A materialized LoRA merge would use another route and a fresh destination:

```sh
# LORA EXPORT WORKFLOW ONLY; unexecuted; quality UNKNOWN.
llama-export-lora -m ./staging/base.gguf \
  --lora ./staging/lora-adapter.gguf -o ./derived/candidate.gguf
```

**Inferred limits:** the destination must not exist or alias an input; qualify the exact exporter's restrictions and output precision, verify expected results, and carry complete notices/manifest before treating it as shareable. Do not promise equal file size, quantization or speed. A **sparse exact replacement** cannot be passed to `--lora` or this exporter. It needs its own qualified GGUF mapping or a separately verified prepared output; without one, refuse application.

**Inferred evidence gate:** runtime-adapter, exported adapter and sparse/prepared-output treatments each need their own baseline and comparison at exact digests/profiles. Register failures and regressions alongside positives. Loading a GGUF does not establish improvement; every example above retains quality UNKNOWN.

## Stop rather than guess

Refuse application when identities/sidecars are missing, a digest or size mismatches, a source/version/route is unsupported, rights are unresolved or an output cannot be safely created. Preserve inputs and partial-work status after interruption. Keep UNKNOWN discovery records visible with the missing fact and owner. A future matched evaluation must freeze its own protocol, denominator, failure policy and guardrails before claiming support; neither workflow transfers evidence to another artifact or profile.
