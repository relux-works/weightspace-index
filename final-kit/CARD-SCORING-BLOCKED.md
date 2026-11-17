# Retired internal diagnostic witness — not a Mac pilot gate

The historical witness below concerns an obsolete proposed card policy, **not RikiyaT's evaluation**. It is retained solely to explain the internal diagnostic task and comparator refusal. No 12-item owner smoke, manual worksheet or exploratory hand-run is required or used as a reference. In RikiyaT's actual code a correct boxed cap-hit can still be correct; cut_share is separate. See RIKIYAT-REPRODUCTION.md for the current source-pinned decisions.

# BLOCKED: fixed scoring cannot be implemented by this YAML alone

The existing `minerva_math500` task is present in 0.4.13. Its inherited prompt is `Problem:\n{problem}\n\nSolution:` with four fixed demonstrations. Its defaults are greedy, stop at `Problem:`, with `exact_match` and `math_verify` means. Dataset: `HuggingFaceH4/MATH-500`, default/test. See SOURCES D/E and the installed task listing.

The single kit YAML reuses **only installed upstream** Minerva `process_docs` (reference extraction and normalization), `process_results` (both offline diagnostic metrics), and the mean aggregations. No upstream prompt or few-shot examples are reused; no response filter is added (the default `none`/take-first path applies). Configuration changes the dataset revision, zero-shot prompt, suffix, stop list, sampling and output cap. No custom Python task/scorer/backend is supplied.

These config changes do **not** implement the retired proposed grading. Source inspection and executed numeric controls prove:

| Response, reference 2 | exact_match | math_verify | Fixed-policy expectation |
|---|---:|---:|---:|
| `\boxed{2}`, normal stop | 0 | 1 | 1 |
| `\boxed{3}`, normal stop | 0 | 0 | 0 |
| `2`, unboxed | 0 | 1 | 0 |
| `\boxed{2}`, finish_reason=length | 0 | 1 | 0 |

`exact_match` looks for Minerva's Final Answer wording; `math_verify` accepts unboxed numerical answers. More fundamentally, the chat backend returns `message.content` alone and discards `finish_reason` and `usage` before scorer execution. A capped answer with a correct box is indistinguishable from the same text with a normal stop. A regex filter or metric rename cannot recover discarded metadata. String-only strict matching would also need robust final balanced-box extraction and agreed math normalization.

**Marked implementation stub — deliberately unimplemented:** preserve response termination/usage, then grade only the final balanced `\boxed{...}` against reference, with every cap-hit wrong. This needs an upstream supported solution or an explicitly authorized extension beyond config. No workaround library or custom Python has been created.

Every bundle is `protocol_qualified:false`; a mechanically complete run is `complete-unqualified`. Both metric values are diagnostics, never accepted MATH-500 scores. G5 currently FAILS. G1–G4/G6 may still test transport/identity/durability mechanics. A real long-generation trace is also needed to determine whether vLLM returns reasoning in content or separately in reasoning_content. Do not change the fixed profile to make a gate pass.
