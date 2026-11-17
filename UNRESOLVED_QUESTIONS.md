# Open publication decisions

- How should dual forms be added to an already published version? The current immutable-version CI rule rejects editing its version.json. The prepared v1 retrofit requires an explicit migration decision; publishing a new version is the alternative. Source replacement bytes and digest are unchanged.
- Universal dual publication needs conversion coverage for nonmatrix/non-BF16 tensors, numeric differences outside the current F64/F32 representation, LoRA-source reverse derivation, and multi-source bundles. These gaps cannot be silently exempted by the tool.
- A future contract revision/migration should make the universal requirement explicit when those paths exist. Contract v1 still permits historical versions without derived forms. Canonical adapter/report pairs have stronger consistency checks.
- Runtime adapter naming/configuration, merge semantics and kernels require separate qualification; PEFT/llama.cpp/vLLM loading remains UNQUALIFIED. No runtime or behavioral result is inferred from numerical derivation.
