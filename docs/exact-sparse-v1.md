# Exact sparse replacement convention v1

Status: **Agreed wsi-sparse-exact v1 (safetensors selected-column), 2026-11-10.** This is the canonical lossless payload convention. The dual-track policy keeps runtime LoRA derivatives as separate derived treatments. Do not zip the lossless delta: preserve its direct mmap/range layout. Agreement on the convention is not a model-runtime support or quality claim. The catalog artifact format remains safetensors, within the unchanged published catalog schema.

## Scope and representations

The operation is `replace-bits`: assign each addressed element its supplied stored bit pattern. There is no subtraction, addition, floating-point multiplication, scale, alpha, quantization or dtype conversion during application. The exact base tensor dtype and shape must match. This preserves signed zeros and NaN payloads rather than normalizing their numerical values.

A safetensors file holds typed index arrays, unsigned bit arrays, and two required metadata strings: `wsi.sparse.version` is `1`; `wsi.sparse.manifest` is one JSON object. Safetensors syntax, endianness and contiguous offset rules come from the [existing container specification](https://github.com/huggingface/safetensors#format), read 2026-11-10. Metadata values remain strings as required by that container. This convention supplies the operation and binding semantics the container itself does not define.

The first profile is limited to: two-dimensional BF16 tensors in the original safetensors checkpoint coordinate system, complete selected-column replacement, little-endian U32 column indices and U16 bit patterns. The offline qualification covers only this profile and this checkpoint pair; no general applicator is distributed here. Corpus releases with selected rows, flat sparse coordinates, F32 directions, 1D biases or packed GGUF updates are not silently admitted through that profile.

Future profiles can retain the same container and define `flat` or `rows` indexing and U32 bit storage for F32. They require a separately specified compatibility table and qualification. F32 directions, additive sidecars, low-rank factors, new tensors and template edits remain distinct operations or treatment kinds. Expanding a direction-derived edit into stored replacements requires naming its base and reproducing the publisher's arithmetic first; this convention cannot preserve an arbitrary floating-point recipe by renaming it a replacement.

## Required manifest

The v1 manifest contains exactly these top-level fields; unknown operational fields refuse interpretation:

| Field | v1 meaning |
|---|---|
| `format` | `wsi-sparse-exact` |
| `version` | integer `1`, matching the metadata version string |
| `operation` | `replace-bits` |
| `byte_order` | `little` |
| `index_order` | `strictly-increasing-row-major-C`: selected columns are sorted; expanded element coordinates traverse rows, then selected columns |
| `duplicates` | `reject` |
| `base_sha256`, `base_size_bytes`, `base_revision` | complete original checkpoint file SHA-256, byte count and immutable source revision; whole-file hashing is mandatory before use |
| `source_repo`, `source_revision`, `source_path`, `source_sha256` | compact edit's publisher/repository, immutable revision, artifact path and measured complete-file SHA-256; source links alone are not locators or permission |
| `edit_license`, `changes` | applicable payload license identifier and an explicit transcode/modification notice |
| `targets` | ordered array of unique named target records |

Each target record contains `name`, `dtype`, `shape`, `indices`, `values`, `changed_count`, `index_layout`, `index_count`, `replacement_count`, `before_sha256`, and `after_sha256`. Array names in `indices` and `values` are unique across targets and refer to exactly the expected safetensors arrays; no unreferenced arrays are admitted. The dtype is `BF16`; `index_layout` is `columns`; shape is `[rows,cols]`. Both shape values are positive and all products must be checked for overflow before allocation.

For a target with `k` selected columns, its index array has U32 dtype and shape `[k]`, and its value array has U16 dtype and shape `[rows,k]` in C order. Index values are strictly increasing and less than `cols`. Position `[i,j]` replaces base tensor `[i,indices[j]]`. `index_count=k`; `replacement_count=rows*k`; `changed_count` is the number of addressed values actually different from the original base bits. It must be recomputed after base binding. A complete selected-column slab may contain no-op values; those positions do not increase `changed_count`.

The per-target hashes cover only that tensor's contiguous stored bytes, without its header, after exact dtype/shape validation. These hashes are explicitly a different domain from whole-file hashes. `before_sha256` must match the original base tensor; `after_sha256` must match the prepared replacement result. A source-archive SHA-256 and a serialized output SHA-256 identify other byte domains and cannot substitute for either check.

U32 indices cover dimensions up to 2^32; larger dimensions refuse this profile. There is no guessed U64 upgrade, signed reinterpretation or negative indexing. All numerical bounds apply to decoded unsigned integers, independent of a programming language's default integer width. Multi-byte data uses little endian. Index arrays encode coordinates, never dequantized offsets in a GGUF block.

## Safe application and failure behavior

Before creating an output, verify the payload's expected complete-file SHA-256 and byte count from its admitted catalog artifact. This detects transport mutation, but it is not proof that a newly generated file has correct semantics. Then bound and validate the safetensors header and metadata; reject duplicate JSON keys, unknown versions, unsupported profiles, unexpected fields, missing arrays, dtype/shape mismatches, integer overflow, invalid extents, gaps/overlaps, and duplicate or out-of-range columns. A profile is an allowlist of known valid forms, not a set of filename exclusions.

Independently hash the complete base checkpoint against the manifest's expected identity before any replacement. Validate every target and prepare its replacement bytes before writing. Any target failure refuses the entire operation. Preserve all untargeted tensors, their dtypes and bytes. Write a separate task-owned temporary file with exclusive creation; synchronize it, publish only to a previously absent destination, and reopen it for all-tensor comparison. Never patch the user's original file in place or overwrite an existing destination. The harness uses an exclusive temporary file and a non-overwriting hard-link publication step on its local filesystem; it does not claim crash recovery, cancellation support, cross-filesystem publication or portable filesystem durability.

A base file hash mismatch refuses even if all edited tensor hashes happen to match: an untouched tensor may have changed. Model configuration, tokenizer, template and runtime context are separate profile identities. This tensor-only payload must not claim that they have been verified by a weight-file hash. Unknown context remains explicit and blocks any operational route requiring it.

Within one payload, duplicate target names and any overlapping addressed coordinates refuse. This single-payload profile does not define composition of distinct overlapping payloads. Such a route requires a separately agreed order/conflict contract and retained evidence. No implicit additive interpretation exists.

## Acceptance contract and limits

Positive controls: named column expansion, unsigned bit preservation including signed zero/NaN fixture values, recognized NPY field-order/quote compatibility, full stored-array coverage, immutable base/source identities, and zero-change application. Negative controls: duplicate indices/targets/JSON keys, descending or out-of-range indices, wrong base/source/version/shape/dtype, unknown operational metadata, truncated/mutated files and an existing destination. The real-payload acceptance requires reopening the saved transcode in a new process, decoding it, removing only base-identical addressed values, comparing the remaining coordinates/bits with the original qualified edit, applying it, and comparing every tensor against both the saved qualified application and the independently downloaded pinned reference.

Freshly reacquired original source files and the pinned checkpoint pair match the recorded complete-file sizes and digests; compact originals also match pinned Git blob identities before parsing. Original application and application of the untouched 264,800-byte payload each reproduce all 488 stored tensors against the pinned reference with zero mismatches and maximum ULP 0. Both targets have 64 selected columns; the payload addresses 131,072 positions containing 130,188 actual changes and 884 no-ops. Removing only base-identical positions reproduces the original edit's complete coordinate/bit set. Re-encoding from the fresh inputs produces byte-identical payload bytes; publication retains the original release bytes. See the [entry byte evidence](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/evidence/byte-qualification.json), [source round-trip](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/evidence/round-trip.json) and [payload round-trip](../models/RikiyaT/Qwen3.5-0.8B-weight-space-programming/patches/qwen35-p50c-exact/evidence/payload-round-trip.json).

The immutable payload SHA-256 is 527139e3ec7870416c6d8e22026d884273cd9221444041114bea44edabd9fedc. Its descriptive changes string retains the original historical words “Proposed format, not adopted.” Preserving that historical string avoids changing the qualified bytes; current agreement status is defined by this v1 document and the catalog, not by that non-operational sentence. The manifest's operation, version, identities and replacement semantics remain unchanged.

The harness also executes synthetic positive/negative admission and preservation controls, plus real original-archive corruption, swapped-name and full wrong-base controls. Those controls do not qualify runtime model loading, quantization survival, quality, or any other publisher's release. A container reader recognizing the arrays is not a model-runtime adapter loader.

## Attribution and changes

The original edit remains subject to its CC BY 4.0 terms and applicable base-weight notices. Preserve authors, article, immutable source links, license identification and modification history in the catalog record and distributed notice. A column-slab transcode adds base-identical addressed values and exact whole-base binding; after removing those no-ops, its changed coordinates and new BF16 bits must match the original archive. It does not modify the qualified final tensor values. No upstream script implementation is copied into the harness. Rights in new code/documentation and the index repository's own license remain separate owner decisions.
