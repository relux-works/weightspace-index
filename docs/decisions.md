# Design decisions

Decisions recorded 2026-11-10. The first [registered entry](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/README.md) is byte-qualified and has quality **UNKNOWN**.

## Registry first; application tooling is optional

An entry should be useful through ordinary Git, a browser, a file download and a SHA-256 tool. The catalog identifies the exact base, payload, locations, component licenses, treatment and missing evidence. A separately qualified applicator can consume those facts later. Discovery must remain usable without that applicator, and discovery must never imply application eligibility.

## Two tracks with different promises

The canonical lossless payload is **wsi-sparse-exact v1**, using safetensors selected-column storage. Its operation replaces stored bits on an exact base. The parallel LoRA track is for **derived treatments** that runtimes can consume through their existing LoRA interfaces, including llama.cpp and vLLM CLI LoRA. Avoiding a separate patch applicator is the LoRA track's value; it is not a reason to call an approximation lossless. Runtime documentation establishes general interfaces, not compatibility of an unpublished derivative. See the [workflow interface references](workflows.md) and the [v1 contract](exact-sparse-v1.md).

Exact rank and energy concentration answer different questions. Each of the two recorded difference matrices has **exact rank 64**. The rank-1 energy synopsis is **99.9994%** for layer 15; the retained measurement table reports **99.999453994%** for layer 15 and **99.999138167%** for layer 19. These recorded figures do not make either difference exactly rank 1. The [spectrum data](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/data/spectrum.json) retains the exact modular-rank evidence, all computed singular values and the arithmetic convention.

The published exact payload is **264,800 bytes**. The recorded rank-1 F16 factor-array budget is **18,432 bytes for both targets**, the roughly **18 KB** shorthand; this excludes an adapter container, configuration and notices. Under the recorded reconstruction recipe, rank-1 F16 factors leave **6,700** and **6,464** differing BF16 values in layers 15 and 19 respectively: roughly **6k per target**, not a total for both. Rank-1 F32 factors leave **5,888** and **6,156** differences and use **36,864 raw factor bytes for both targets**. These are recorded numerical reconstruction results, not runtime or behavioral measurements. The [error table](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/data/low-rank-errors.json) preserves the values without a new SVD or a new rounding step. No LoRA adapter is distributed in this snapshot; any derivative needs its own identity, route, attribution and assessment.

## Container choice and direct access

| Recorded representation | Complete payload and required binding bytes | Tradeoff |
|---|---:|---|
| Published safetensors selected-column v1 | 264,800 | Canonical lossless payload; named typed arrays, explicit bindings, no outer compression. |
| Compressed NPZ selected-column with embedded binding | 210,997 | Smaller recorded comparison; ZIP Deflate requires decompression before typed array access. |
| GGUF-carried selected-column comparison | 264,864 | Recorded raw-bit container experiment; needs a sparse application convention distinct from GGUF model or LoRA loading. |

The [recorded container comparison](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/data/container-comparison.json) also distinguishes the **264,792-byte** safetensors comparison serialization from the published **264,800-byte** release. Equivalent decoded arrays do not make those file identities interchangeable. These measurements include the respective mandatory binding data, exclude shared notices, and are not repeated speed or memory benchmarks.

Safetensors selected-column is selected for explicit operation/base binding, one-file custody and direct access to contiguous typed regions. It is not the smallest measured container. **Do not zip or otherwise compress the canonical lossless delta:** its file layout must remain directly accessible for mmap/range readers. Mmap avoids an obligatory eager tensor copy in a suitable reader; it does not guarantee bounded resident memory, zero physical I/O, safe application, or runtime support. Compressed NPZ remains a useful comparison rather than a forbidden file type. GGUF-carried sparse arrays are not automatically a GGUF LoRA adapter, and packed-quantized coordinate editing remains outside this profile.

## Two storage modes; one digest identity

The catalog admits an in-Git `index-path` in a pinned index snapshot and an external `hf` locator with repository type, case-preserving name, full immutable revision and relative path. Both bind the same complete-file SHA-256 and byte count; multiple locators for an artifact must name identical bytes. The first exact payload is in Git at its catalog locator. The base and patched reference are linked at immutable HF revisions. No HF mirror of the new payload is claimed; a future external copy needs an independently checked identical digest before a locator is added. Source citations, including the original compact edit's GitHub links, do not silently become operational fetch locators.

## Storage and runtime are separate

A container describes stored bytes. The replacement operation, coordinate system, full base hash, target dtypes/shapes and before/after hashes define application. A runtime adapter interface is a separate contract. Byte equality against the pinned reference covers stored tensors only; configuration, tokenizer/template, model loading, quantization survival, behavior and performance remain unqualified. The exact path needs a separately qualified utility; the derived LoRA path uses a runtime's separately qualified adapter route. Neither can inherit the other's evidence by sharing a model name.

## Implementation language

**Rust is selected for future application tooling.** It allows direct use of the [official safetensors core and borrowed views](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/safetensors/src/tensor.rs), with the documented caller-provided mmap route and alignment with the [official Python binding](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/bindings/python/Cargo.toml). The earlier Go hybrid proposal was conditional on required format, memory, UI and delivery qualification. Giving safetensors equal weight made direct official implementation reuse decisive: the inspected [community Go reader](https://github.com/nlpodyssey/safetensors/blob/bfb01cc25f7c9e2170aafdec0587c8e6f676021d/safetensors.go) requires a complete-file byte slice, so an ordinary `os.ReadFile` route allocates a whole shard, and its dtype coverage is narrower. That whole-file reader route was rejected for the intended tool; this is not a claim that Go cannot mmap or stream writes, nor a measured Rust speed or binary-size win. The selected shape is one Rust application/core, with separately qualified external numerical mergers when required, rather than adding a Go wrapper around a new Rust engine. This publication contains data and documentation; no Rust applicator is released here.

The recorded source inspection identifies the [official core manifest](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/safetensors/Cargo.toml) as **0.9.0-dev.0**, requiring **Rust 1.86**; the inspected [Go GGUF writer candidate](https://github.com/abrander/gguf/blob/b2659dbc89578649e999e0fce964f2777b326231/go.mod) requires **Go 1.27**. These are recorded source/build facts, not installed application requirements or passed tool qualification. Equivalent whole-tool binary size, speed and peak memory remain **UNKNOWN**; none is inferred from a language or library version.

## Evidence follows each exact artifact

Every artifact requires its own byte count, full digest, digest status and source. A retained verification reference must resolve to evidence for that exact file and byte domain. Whole-checkpoint hashes, tensor hashes, original archive hashes, transcode hashes and serialized output hashes are different domains. The first entry's [byte qualification](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/evidence/byte-qualification.json) records fresh input identities, the original edit, decoded payload reconciliation and zero-ULP stored-tensor results. The output keeps the base header, so tensor equality does not mean its whole-file hash equals the publisher's patched file.

**UNKNOWN** is a reasoned evidence gap, not a score, rejection, support verdict or certification. This entry has no evaluation runs or comparisons and must not be listed as supported. The dual-use acceptance policy remains an open decision: edits that remove behavior require an explicit policy before any supported listing. Registering this reasoning-loop math edit does not settle that policy or establish efficacy.

## Attribution and license layers

Base Qwen weights retain applicable **Apache-2.0** duties. The compact edit and its lossless transcode retain **CC BY 4.0** credit, source links, disclaimers and change marking, including the added base-identical column positions and identity bindings. Publisher scripts have an MIT allocation in their source license, but **no publisher script implementation is reused or redistributed** here. The source allocation is retained as context; MIT does not replace the edit's CC BY terms. See the [entry attribution and component notices](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/notices/ATTRIBUTION.md). The index repository's own license remains undecided and no repository LICENSE file is added. A representation change does not erase upstream obligations or imply endorsement.
