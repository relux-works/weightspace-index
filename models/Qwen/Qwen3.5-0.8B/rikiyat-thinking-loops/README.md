# Qwen3.5-0.8B, edited by an LLM with no training (Weight-Space Programming)

The title is the [author's published card headline](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md). RikiyaT frames the edit around the model's “thinking loops” in that card. This entry records the author's edit in a maintainer-produced container; see [metadata](patch.json) and [version 1](v1/version.json).

## Base model

The target is the exact checkpoint of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) pinned in [base.json](../base.json). Architecture observation from that pinned checkpoint's tensor headers: it contains language-model and visual tensor namespaces, including language-model MLP down-projection tensors in layers 15 and 19, each BF16 with shape [1024, 3584]. This describes the observed tensor structure, without assigning a precise external architecture name. The checkpoint has 488 tensors (452 BF16, 36 F32); the `bf16` profile identifies this exact weight file, not a promise that every tensor has the same dtype. Configuration and tokenizer files are outside this checkpoint-only profile.

## Lineage and representation

| Published input | Revision | Path | Bytes | SHA-256 |
|---|---|---|---:|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `model.safetensors-00001-of-00001.safetensors` | 1746942600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/tree/d1671a178705bcab7644806d23c535517f35723b) | `d1671a178705bcab7644806d23c535517f35723b` | `model.safetensors-00001-of-00001.safetensors` | 1746943032 | `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553` |

Maintainers built the canonical representation with Weightspace Index's `weightspace-tools` 0.1.0 at source commit `d15c7087a344cc65fd5cdb5b80298f01d711118f`, using `weightspace replacement` on these two published checkpoints. It is the version's `source` under the [canonical-representation rule](../../../../docs/structure.md#patch-forms), rather than an author-published patch file. RikiyaT retains edit authorship; the submitter supplied the representation.

The construction was hash-gated by the [author's manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json), projected to the tool's strict five-field tensor format without changing hashes or shapes (dtype spelling `bfloat16` → `BF16`). The sealed byte-validation evidence reproduced the replacement and verified an apply round trip: 488/488 tensor descriptors and SHA-256 digests matched the edited checkpoint; the other 486 tensors matched the base. Exactly 130,188 changed coordinates were represented in the two tensors. The new container preserves replacement values and sparse `#old` base values. Whole-checkpoint container bytes need not match: apply retains the base container layout. These are byte-provenance checks only.

## Apply

Build/install `weightspace` using the [tool guide](../../../../tool/README.md). Verify the base checkpoint's size and SHA-256 against [base.json](../base.json) and the patch against [v1/version.json](v1/version.json). The current CLI requires one root `model.safetensors`; stage a copy of the pinned checkpoint under that name in a fresh directory:

```sh
mkdir base-for-apply
cp /path/to/verified/model.safetensors-00001-of-00001.safetensors base-for-apply/model.safetensors
mkdir -p scratch-snapshots
weightspace apply \
  --patch models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/v1/files/replacements.safetensors \
  --model base-for-apply \
  --out patched-checkpoint \
  --temp-dir scratch-snapshots
```

Run from the index checkout with a new output path and enough disk for the checkpoint copy and private snapshots. This writes a checkpoint copy; a full runtime setup also requires the base model's configuration and tokenizer.

## Name, license and evidence

The slug `rikiyat-thinking-loops` uses the author's namespace `RikiyaT`, lowercased, and the exact phrase “thinking loops”, slugified word-for-word as `thinking-loops`, from the [pinned card](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md). The phrase describes the traced failure mechanism; the card's stated goal is to make the model “better at math”.

The base and edited checkpoint are published under Apache-2.0. The original compact edit's data are published under CC BY 4.0. Both layers travel with this representation: [attribution and change marking](notices/ATTRIBUTION.md), [Apache-2.0](notices/Apache-2.0.txt), and [CC BY 4.0](notices/CC-BY-4.0.txt).

**no benchmarks submitted**. No inference was run for this entry. It contains no run, comparison or assessment records and makes no behavioral efficacy claim.

## Derived LoRA publication

Version 1 also lists `adapter.safetensors` and `lora-report.json` as derived files from the unchanged replacement source. Weightspace tools 0.2.0 built the per-tensor `selector-exact` adapter offline from saved #old/#new values and the [1024, 3584] matrix shapes recorded above; no base weights were downloaded for this derivation. To reproduce, write a JSON object mapping both complete tensor names (layers 15 and 19) to `[1024, 3584]`, then run:

```sh
weightspace derive-lora --patch v1/files/replacements.safetensors \
  --tensor-shapes tensor-shapes.json --method selector-exact \
  --factor-dtype f32 --out derived-lora
```

Each tensor uses rank 64. The adapter is 2,359,816 bytes (2,359,296 raw F32 factor bytes), compared with 1,563,032 bytes for the exact replacement; conversion is not conditional on compression. The report records zero frozen-BF16 bake mismatches on 65,260 and 64,928 saved coordinates respectively. F32 delta Frobenius errors are approximately 2.33e-10 and 8.73e-11; algebraic selector exactness does not mean exact stored delta. Unseen base values were not verified. See the report's precise values, verification scopes, digests and frozen arithmetic rule.

Names append literal `.lora_A` / `.lora_B`, with scale 1; no adapter_config is supplied. **PEFT/llama.cpp/vLLM loading is UNQUALIFIED.** No inference or benchmark was run for this derived treatment, and the source study's claims do not transfer to it.
