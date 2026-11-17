# Patch formats

Every patch version has one **source**, in one of two forms. A new form is a change to [structure.md](structure.md#patch-forms), never something a submission introduces.

A source file may be a maintainer-produced canonical representation
of an author-published edit, under the lineage and validation rules
in [structure.md](structure.md#patch-forms). It is the version's
source for derived files; it is never itself listed as `derived`.

| | Replacement | LoRA |
|---|---|---|
| `kind` | `replacement` | `lora` |
| Files | One or more `.safetensors` files keyed by the base model's own tensor names | As published: PEFT `adapter_model.safetensors` with `adapter_config.json`, or one GGUF LoRA |
| Values | New values in the target profile's dtype | Low-rank factors and their scaling |
| Applied by | Assignment into a copy of the target profile | The runtime: loaded as an adapter, or merged with the runtime's own tool |
| Exactness | Bit-exact | As exact as the runtime's arithmetic |
| Works on | Only the target profile, which is unquantized | Any profile the runtime can attach it to, including GGUF quants |

## Replacement files

A changed tensor is stored one of two ways:

- **Full:** the whole new tensor under its own name, for example `model.layers.0.mlp.down_proj.weight`.
- **Sparse:** three tensors for a tensor that changes in a few places:
  - `<name>#idx`: flat indices into the tensor, `int64`, sorted and unique;
  - `<name>#new`: the new values, in the tensor's dtype, one per index;
  - `<name>#old`: the values they replace, same shape and dtype.

A tensor is either full or sparse, never both. Values are stored, never differences: assigning a value is exact, while adding a difference in a low-precision dtype can round. `#old`, together with explicitly supplied original matrix shapes, lets offline checks rebuild derived sparse selectors without downloading the base, and lets a user confirm their copy matches before writing.

## Derived files

The publication expectation is to provide replacement and LoRA forms together, keeping one source form and listing the other as `role: derived`, with `derived_from` source digests and tool name/version. Historical contract-v1 records without dual files remain readable. See [coverage and contract boundaries](structure.md#patch-forms).

- A **LoRA derived from a replacement**: per-tensor `selector-exact` handles sparse and whole BF16 matrices without a size or sparsity gate. It emits `adapter.safetensors` plus `lora-report.json`. Exactness is the algebraic construction before F32 storage; the report measures delta loss and a frozen BF16 bake. Saved `#old` values plus declared matrix shapes permit offline sparse rebuilding, with metrics limited to saved coordinates. Runtime loading, including use with quants, remains **UNQUALIFIED**; no runtime config is emitted. Study claims never transfer to this treatment.
- A **GGUF conversion of a PEFT LoRA**, made with llama.cpp's own converter.

## What is not a patch

- **A complete model.** A fine-tune or merge that publishes every tensor is a base of its own, with its own folder.
- **A quantized copy of a patched model.** The client never quantizes or converts; quants get the LoRA.
- **Pickle formats** (`.bin`, `.pt`, `.pth`, `.pkl`) and code of any kind: loading them can execute code.
