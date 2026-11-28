# Inkblot-2-mini (Loop-Control Weight Edit for Inkling-2-Mini)

The title and slug come directly from the [author's published model card](https://huggingface.co/Ex0bit/Inkblot-2-mini/blob/fb0b924f46a74331e34ce4f4288177f9dc0383c8/README.md) (`# Inkblot-2-mini`) under the [Ex0bit](https://huggingface.co/Ex0bit) Hugging Face namespace. The edit is the work of Eric E. (Tensorbend.ai Research), described in [`eelbaz/Recursive-Self-Improving-Loopers`](https://github.com/eelbaz/Recursive-Self-Improving-Loopers). This entry records the author's edit as a maintainer-produced canonical replacement container; see [metadata](patch.json) and [version 1](v1/version.json).

## Base model

The target is the exact pinned checkpoint of [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195) in [base.json](../base.json). Architecture observation from the pinned checkpoint headers: `Inkling2LoopForConditionalGeneration` (24-layer looped multimodal MoE, 26.39B total / 3.4B active parameters; middle Layers 6–17 repeat 3 times by default). `model-00001-of-00001.safetensors` holds 1,244 tensors across mixed precision (`994 BF16`, `154 F32`, `48 F8_E4M3`, `48 U8` NVFP4/MXFP8 experts), alongside an unindexed `mtp.safetensors` sidecar (multi-token prediction head). Every companion file (`config.json`, `chat_template.jinja`, `hf_quant_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `mtp.safetensors`) is byte-identical between base and edited repositories.

## Lineage and representation

| Published input | Revision | Path | Bytes | SHA-256 |
|---|---|---|---:|---|
| `thinkingmachines/Inkling-2-Mini` | `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` | `model-00001-of-00001.safetensors` | `19526019168` | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| [`Ex0bit/Inkblot-2-mini`](https://huggingface.co/Ex0bit/Inkblot-2-mini/tree/fb0b924f46a74331e34ce4f4288177f9dc0383c8) | `fb0b924f46a74331e34ce4f4288177f9dc0383c8` | `model-00001-of-00001.safetensors` | `19526019168` | `59312c8841614cff95b6531cd259235d0175b09220aed84eddeae8f3ed354b58` |
| [`reluxworks/Inkling-2-Mini-ex0bit-inkblot-2-mini`](https://huggingface.co/reluxworks/Inkling-2-Mini-ex0bit-inkblot-2-mini/tree/f1a1b8e7a51eb15d7117af9ef5396d59a17f10b2) | `f1a1b8e7a51eb15d7117af9ef5396d59a17f10b2` | `replacements.safetensors` | `1003051184` | `fa671bf1e22449893f1563112bc95cb0c5fd5f20003abe9a9185f419e11bd756` |

Maintainers built the canonical replacement file with `weightspace-tools` 0.2.0 (`weightspace replacement`) from the two pinned checkpoints above. It is the version's `source` under the [canonical-representation rule](../../../../docs/structure.md#patch-forms), hosted on Hugging Face because the file (`956.58 MiB`) exceeds GitHub's 100 MB per-file cap while reducing download size by ~19× relative to the `18.18 GiB` full weights file.

Exactly **327 of 1,244 tensors** differ (`39 F32`, `288 BF16`; `501,285,796` changed entries): 325 tensors are stored in full form and 2 short-convolution tensors (`model.llm.layers.11.attn_sconv.weight` with 1,135 changed elements and `model.llm.layers.23.attn_sconv.weight` with 1,077 changed elements) are stored in sparse `#idx`/`#old`/`#new` form. All 917 remaining tensors — including every quantized MoE expert (`F8_E4M3` / `U8`), vision weight, and `mtp.safetensors` — are untouched.

Round-trip verification with `weightspace apply` against the pinned base checkpoint reproduced **1,244 / 1,244 tensors bit-identically** (`0` mismatches) and reproduced the exact whole-file SHA-256 `59312c8841614cff95b6531cd259235d0175b09220aed84eddeae8f3ed354b58` of `Ex0bit/Inkblot-2-mini/model-00001-of-00001.safetensors`, while preserving `mtp.safetensors` (`7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`) byte-for-byte.

## Apply

Verify the pinned base checkpoint against [base.json](../base.json) and `replacements.safetensors` against [v1/version.json](v1/version.json). Because 325 of the 327 edited tensors are stored in full form, pass the companion `base-tensor-sha256.json` digest map from [`reluxworks/Inkling-2-Mini-ex0bit-inkblot-2-mini`](https://huggingface.co/reluxworks/Inkling-2-Mini-ex0bit-inkblot-2-mini/blob/f1a1b8e7a51eb15d7117af9ef5396d59a17f10b2/base-tensor-sha256.json) so `weightspace apply` verifies every target base tensor before writing:

```sh
weightspace apply \
  --patch /path/to/replacements.safetensors \
  --expect-tensor-sha256-file /path/to/base-tensor-sha256.json \
  --model /path/to/thinkingmachines-Inkling-2-Mini \
  --out /path/to/Inkblot-2-mini-patched \
  --temp-dir /path/to/scratch-snapshots
```

The output directory is a drop-in checkpoint usable at default 3-pass inference with standard `Inkling-2-Mini` runtimes (`transformers`, `vLLM`, `SGLang`). Overriding loop depth to 4 passes (`LoopConfig(num_passes=4)`, where the author reports `70.3%` on `DeepScaleR-Test` with `0%` repeat loops) additionally requires the author's [`rse_looper`](https://github.com/eelbaz/Recursive-Self-Improving-Loopers) forward wrapper at runtime.

## Name, license and evidence

The slug `ex0bit-inkblot-2-mini` combines the author's Hugging Face namespace `Ex0bit` (lowercased) with the edit's own published name `Inkblot-2-mini` from the [pinned model card](https://huggingface.co/Ex0bit/Inkblot-2-mini/blob/fb0b924f46a74331e34ce4f4288177f9dc0383c8/README.md).

Both the base checkpoint and edited checkpoint are published under `Apache-2.0`; see [notices/ATTRIBUTION.md](notices/ATTRIBUTION.md) and [notices/Apache-2.0.txt](notices/Apache-2.0.txt).

Version 1 currently records author-reported results only (`Reported`), split cleanly between [default 3-pass inference](v1/claims/inkblot-3pass.json) and [4-pass `rse_looper` inference vs. default 3-pass base](v1/claims/inkblot-4pass.json). No independent Weightspace Index benchmark runs have been filed yet.

## Derived LoRA coverage boundary

No derived `adapter.safetensors` is published for version 1. Of the 327 edited tensors, 252 are 2-D `BF16` matrices, while 75 fall outside `weightspace derive-lora --method selector-exact`'s current [coverage boundary](../../../../docs/structure.md#patch-forms) (`36` 3-D `BF16` short-convolution weights, `34` 1-D `F32` query vectors, and `5` 2-D `F32` tensors). Under `contract_version: 1`, omitting derived files is valid when conversion coverage is incomplete.
