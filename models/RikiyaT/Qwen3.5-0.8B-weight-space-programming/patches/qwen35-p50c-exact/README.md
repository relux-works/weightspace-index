See the warning in the [root README](../../../../../README.md) and [DISCLAIMER.md](../../../../../DISCLAIMER.md).

# Qwen3.5-0.8B p50c exact replacement

**One registered, byte-qualified entry; quality UNKNOWN. Not listed as supported.** The authoritative records are in [catalog.json](../../../../../catalog.json): entry `qwen35-p50c-exact`, treatment `qwen35-p50c-sparse-v1`, assessment `qwen35-p50c-unknown`. This page explains those records; it is not a second score registry.

The source presents p50c as a reasoning-loop math edit. Original research: **Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab**, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/). Study results are external claims and are not reproduced or inherited by this entry or any future derived treatment. No efficacy claim is made here.

## Exact identities

| Component | Immutable revision | Bytes | Complete-file SHA-256 |
|---|---|---:|---|
| [Base: Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | `2fc06364715b967f1860aea9cf38778875588b17` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| [Patched reference](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/tree/d1671a178705bcab7644806d23c535517f35723b) | `d1671a178705bcab7644806d23c535517f35723b` | 1,746,943,032 | `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553` |
| [Original edit.npz](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/edit.npz) | `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` | 387,619 | `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7` |
| [Original manifest.json](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json) | same source revision | 1,326 | `83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225` |
| [Published exact payload](../../../../../payloads/qwen35-p50c-exact/v1/qwen35-0.8b-p50c.safetensors) | caller must pin the index snapshot | 264,800 | `527139e3ec7870416c6d8e22026d884273cd9221444041114bea44edabd9fedc` |

Both checkpoint paths are `model.safetensors-00001-of-00001.safetensors`. The compact-source paths are `exp8_math/runs/p50c/edit/{edit.npz,manifest.json}`. The checkpoint hashes match the pinned advertised LFS identities and fresh complete-file measurements. Compact-source SHA-256 values are measured identities, not publisher-advertised SHA-256 claims; the original files also match their pinned Git blob identities, recorded in [byte evidence](evidence/byte-qualification.json).

## What the lossless payload contains

The agreed [wsi-sparse-exact v1 convention](../../../../../docs/exact-sparse-v1.md) stores selected-column U32 indices and raw U16 BF16 bits in uncompressed safetensors. It is a stored-bit replacement, not an additive delta or a LoRA adapter. The catalog's format remains `safetensors`; this convention versions application semantics separately.

| Target tensor suffix | Shape / dtype | Selected columns | Addressed positions | Actual changes | Base-identical no-ops |
|---|---|---:|---:|---:|---:|
| `layers.15.mlp.down_proj.weight` | `[1024,3584]` / BF16 | 64 | 65,536 | 65,260 | 276 |
| `layers.19.mlp.down_proj.weight` | `[1024,3584]` / BF16 | 64 | 65,536 | 64,928 | 608 |
| Total | two target tensors | 128 | 131,072 | 130,188 | 884 |

The full target names begin with `model.language_model.`. After removing only base-identical addressed positions, decoded coordinates and replacement bits equal the freshly acquired original edit and the independently pinned checkpoint difference. Both original and published-payload applications reproduce **all 488 stored tensors**, **873,438,784 elements**, with **zero mismatches and maximum ULP 0**; see [original round-trip](evidence/round-trip.json) and [payload round-trip](evidence/payload-round-trip.json). The saved application retains the base header; its whole-file hash differs from the patched reference despite complete tensor equality. That generated file is not a separately published model artifact.

## Spectrum and the derived track

Each exact difference matrix has **rank 64**, supported by its 64 active columns and exact modular-rank lower bounds. Rank-1 energy concentration is a separate numerical claim: the recorded synopsis **99.9994%** refers to layer 15; retained values are **99.999453994%** for layer 15 and **99.999138167%** for layer 19. See [spectrum data](data/spectrum.json), [recorded reconstruction errors](data/low-rank-errors.json), and [decisions](../../../../../docs/decisions.md). Concentration does not make the rank-1 approximation exact or establish usefulness.

The dual-track policy keeps this exact payload canonical and treats tool-free runtime LoRA derivatives as separate treatments with separate bytes, application recipes and assessments. No LoRA artifact is supplied. No runtime adapter, model loading, GGUF/quantized survival, tokenizer/template or configuration qualification follows from this tensor proof. The runtime examples in the inherited scaffold remain workflows, not qualified routes for this entry.

## Acquisition, limits and credit

The payload's authoritative local locator is `payloads/qwen35-p50c-exact/v1/qwen35-0.8b-p50c.safetensors` in a pinned index snapshot. Keep it uncompressed for direct mmap/range access; do not zip the lossless delta. The HF links above identify different components, not alternate bytes for this payload. The index supports in-Git and immutable HF locator modes; no new payload mirror or general downloader is supplied.

Quality, behavioral benefit, runtime compatibility and safety remain **UNKNOWN**. There are no evaluation runs or comparisons. The dual-use acceptance policy is still undecided; it must exist before any behavior-removing entry can be listed as supported. Registration of this math-edit entry does not settle that open policy.

Preserve [attribution and change marking](notices/ATTRIBUTION.md), the [Apache-2.0 base notice](notices/Apache-2.0-Qwen.txt), and the [source license allocation](notices/SOURCE-LICENSE-lm_understand-logs.txt). The compact edit and lossless transcode are CC BY 4.0 with applicable base notices. No publisher script implementation is reused or redistributed. No repository LICENSE is chosen here.
