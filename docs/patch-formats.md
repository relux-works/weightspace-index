# Patch formats

Every patch version has one **source**, in one of two forms. A new form is a change to [structure.md](structure.md#patch-forms), never something a submission introduces.

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

A tensor is either full or sparse, never both. Values are stored, never differences: assigning a value is exact, while adding a difference in a low-precision dtype can round. `#old` lets CI check derived files without downloading the base, and lets a user confirm their copy matches before writing.

## Derived files

The index may build convenience files from a source and list them as `role: derived`, with `derived_from` and the tool that built them. CI rebuilds each one and compares digests; submitters never provide them.

- A **LoRA built from a replacement**, so the patch works on quants and in adapter-only runtimes. It reproduces the replacement up to runtime rounding and stays small only when the edits touch few rows or columns; a replacement of whole tensors gets none.
- A **GGUF conversion of a PEFT LoRA**, made with llama.cpp's own converter.

## What is not a patch

- **A complete model.** A fine-tune or merge that publishes every tensor is a base of its own, with its own folder.
- **A quantized copy of a patched model.** The client never quantizes or converts; quants get the LoRA.
- **Pickle formats** (`.bin`, `.pt`, `.pth`, `.pkl`) and code of any kind: loading them can execute code.
