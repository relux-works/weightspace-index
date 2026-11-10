# Design decisions

Decisions recorded 2026-11-10. The first entry is withdrawn pending base-first restructuring. No entry or payload is currently registered; the generic format and design decisions below remain in effect.

## Registry first; application tooling is optional

An entry should be useful through ordinary Git, a browser, a file download and a SHA-256 tool. The catalog identifies the exact base, payload, locations, component licenses, treatment and missing evidence. A separately qualified applicator can consume those facts later. Discovery must remain usable without that applicator, and discovery must never imply application eligibility.

## Two tracks with different promises

The canonical lossless payload is **wsi-sparse-exact v1**, using safetensors selected-column storage. Its operation replaces stored bits on an exact base. The parallel LoRA track is for **derived treatments** that runtimes can consume through their existing LoRA interfaces, including llama.cpp and vLLM CLI LoRA. Avoiding a separate patch applicator is the LoRA track's value; it is not a reason to call an approximation lossless. Runtime documentation establishes general interfaces, not compatibility of an unpublished derivative. See the [workflow interface references](workflows.md) and the [v1 contract](exact-sparse-v1.md).

Exact rank and energy concentration answer different questions. Energy concentrated in a low-rank approximation does not make an edit exactly low rank. Any derivative needs its own byte identity, route, attribution, reconstruction evidence and assessment. No LoRA adapter is distributed in this snapshot.

## Container choice and direct access

Safetensors selected-column is selected for explicit operation/base binding, one-file custody and direct access to contiguous typed regions. **Do not zip or otherwise compress the canonical lossless delta:** its file layout must remain directly accessible for mmap/range readers. Mmap avoids an obligatory eager tensor copy in a suitable reader; it does not guarantee bounded resident memory, zero physical I/O, safe application, or runtime support. Compressed NPZ remains a useful comparison rather than a forbidden file type. GGUF-carried sparse arrays are not automatically a GGUF LoRA adapter, and packed-quantized coordinate editing remains outside this profile.

## Two storage modes; one digest identity

The catalog admits an in-Git `index-path` in a pinned index snapshot and an external `hf` locator with repository type, case-preserving name, full immutable revision and relative path. Both bind the same complete-file SHA-256 and byte count; multiple locators for an artifact must name identical bytes. No payload is registered in this snapshot. A future external copy needs an independently checked identical digest before a locator is added. Source citations do not silently become operational fetch locators.

## Storage and runtime are separate

A container describes stored bytes. The replacement operation, coordinate system, full base hash, target dtypes/shapes and before/after hashes define application. A runtime adapter interface is a separate contract. Byte equality against the pinned reference covers stored tensors only; configuration, tokenizer/template, model loading, quantization survival, behavior and performance remain unqualified. The exact path needs a separately qualified utility; the derived LoRA path uses a runtime's separately qualified adapter route. Neither can inherit the other's evidence by sharing a model name.

## Implementation language

**Rust is selected for future application tooling.** It allows direct use of the [official safetensors core and borrowed views](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/safetensors/src/tensor.rs), with the documented caller-provided mmap route and alignment with the [official Python binding](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/bindings/python/Cargo.toml). The earlier Go hybrid proposal was conditional on required format, memory, UI and delivery qualification. Giving safetensors equal weight made direct official implementation reuse decisive: the inspected [community Go reader](https://github.com/nlpodyssey/safetensors/blob/bfb01cc25f7c9e2170aafdec0587c8e6f676021d/safetensors.go) requires a complete-file byte slice, so an ordinary `os.ReadFile` route allocates a whole shard, and its dtype coverage is narrower. That whole-file reader route was rejected for the intended tool; this is not a claim that Go cannot mmap or stream writes, nor a measured Rust speed or binary-size win. The selected shape is one Rust application/core, with separately qualified external numerical mergers when required, rather than adding a Go wrapper around a new Rust engine. This publication contains data and documentation; no Rust applicator is released here.

The recorded source inspection identifies the [official core manifest](https://github.com/safetensors/safetensors/blob/9e6f4d6c7af8c63ac791ab1bfdd58d2de02fbaa4/safetensors/Cargo.toml) as **0.9.0-dev.0**, requiring **Rust 1.86**; the inspected [Go GGUF writer candidate](https://github.com/abrander/gguf/blob/b2659dbc89578649e999e0fce964f2777b326231/go.mod) requires **Go 1.27**. These are recorded source/build facts, not installed application requirements or passed tool qualification. Equivalent whole-tool binary size, speed and peak memory remain **UNKNOWN**; none is inferred from a language or library version.

## Evidence follows each exact artifact

Every artifact requires its own byte count, full digest, digest status and source. A retained verification reference must resolve to evidence for that exact file and byte domain. Whole-checkpoint hashes, tensor hashes, original archive hashes, transcode hashes and serialized output hashes are different domains. Tensor equality does not imply whole-file equality when serialized headers differ.

**UNKNOWN** is a reasoned evidence gap, not a score, rejection, support verdict or certification. There are no registered evaluation runs or comparisons. The dual-use acceptance policy remains an open decision: edits that remove behavior require an explicit policy before any supported listing. Format agreement does not settle that policy or establish efficacy.

## Attribution and license layers

Base weights, edits, transcodes, scripts and derived outputs retain their separately applicable terms, attribution, notices and modification history. A representation change does not erase upstream obligations or imply endorsement. No third-party model or patch material is distributed in this snapshot. The index repository's own license remains undecided and no repository LICENSE file is added. See [NOTICE](../NOTICE.md) for the component-level obligations for future entries.
