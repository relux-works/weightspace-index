# Attribution and changes

Original weight edit: Eric E., Tensorbend.ai Research ([published model card](https://huggingface.co/Ex0bit/Inkblot-2-mini/blob/fb0b924f46a74331e34ce4f4288177f9dc0383c8/README.md) on Hugging Face under the [Ex0bit](https://huggingface.co/Ex0bit) namespace; code and technical report in [eelbaz/Recursive-Self-Improving-Loopers](https://github.com/eelbaz/Recursive-Self-Improving-Loopers)). Maintainer canonical representation submitted by `ivanopcode` (`reluxworks`). No endorsement by the author or Thinking Machines Lab, Inc. is implied.

Base weights: `thinkingmachines/Inkling-2-Mini`, Apache-2.0, released by Thinking Machines Lab, Inc. Edited checkpoint: `Ex0bit/Inkblot-2-mini`, Apache-2.0. The full Apache License 2.0 text is preserved in [Apache-2.0.txt](Apache-2.0.txt). No author scripts, Python code, or benchmark datasets are distributed here.

## Pinned published inputs

- Base repository `thinkingmachines/Inkling-2-Mini`, revision `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`:
  - `model-00001-of-00001.safetensors`, `19526019168` bytes, SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`.
  - `mtp.safetensors`, `635801416` bytes, SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`.
- Edited repository `Ex0bit/Inkblot-2-mini`, revision `fb0b924f46a74331e34ce4f4288177f9dc0383c8`:
  - `model-00001-of-00001.safetensors`, `19526019168` bytes, SHA-256 `59312c8841614cff95b6531cd259235d0175b09220aed84eddeae8f3ed354b58`.
- Hosted maintainer canonical replacement repository `reluxworks/Inkling-2-Mini-ex0bit-inkblot-2-mini`, revision `f1a1b8e7a51eb15d7117af9ef5396d59a17f10b2`:
  - `replacements.safetensors`, `1003051184` bytes, SHA-256 `fa671bf1e22449893f1563112bc95cb0c5fd5f20003abe9a9185f419e11bd756`.

## Change marking

On 2026-11-28, maintainers represented the published checkpoint edit as `replacements.safetensors` using `weightspace-tools` 0.2.0 (`weightspace replacement`) on the two pinned checkpoints above. The changes are container/layout and sparse coordinate encoding only (325 full-form tensors and 2 sparse tensors with `#old` values copied from the pinned base). No changed coordinate or replacement value is added, dropped, or altered. The file is a maintainer-produced canonical representation, not an author-published patch file. The original edit's authorship, credit, and Apache-2.0 license obligations remain attributable to its source.
