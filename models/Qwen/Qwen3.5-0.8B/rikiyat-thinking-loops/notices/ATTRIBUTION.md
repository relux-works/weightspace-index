# Attribution and changes

Original weight edit: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, MIT CSAIL ([study](https://rikiyat.github.io/blog/weight-space-programming/); [published card](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md) from Rikiya Takehi's [RikiyaT](https://huggingface.co/RikiyaT) account). Maintainer representation submitted by reluxzion. No endorsement by the authors is implied.

Base weights: Qwen/Qwen3.5-0.8B, Apache-2.0; Copyright 2026 Alibaba Cloud, as retained in [Apache-2.0.txt](Apache-2.0.txt). The [base license](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/LICENSE) and [edited checkpoint license](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/LICENSE) have identical bytes at the pinned revisions.

The original edit's logs/data/networks are licensed under Creative Commons Attribution 4.0 International by the [source repository's license](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE). [CC-BY-4.0.txt](CC-BY-4.0.txt) preserves the full license text from https://creativecommons.org/licenses/by/4.0/legalcode.txt. The entry's license expression records both layers; neither is offered as an alternative to the other. No author scripts, benchmark data or NPZ are distributed here.

## Pinned published inputs

- Base repository `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, path `model.safetensors-00001-of-00001.safetensors`, 1746942600 bytes, SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Edited repository `RikiyaT/Qwen3.5-0.8B-weight-space-programming`, revision `d1671a178705bcab7644806d23c535517f35723b`, path `model.safetensors-00001-of-00001.safetensors`, 1746943032 bytes, SHA-256 `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553`.
- Hash manifest repository `RikiyaT/lm_understand-logs`, revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, path `exp8_math/runs/p50c/edit/manifest.json`, 1326 bytes, SHA-256 `83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225`.

## Change marking

On 2026-11-16, maintainers represented the published checkpoint edit as `replacements.safetensors` using `weightspace-tools` 0.1.0 at `d15c7087a344cc65fd5cdb5b80298f01d711118f`. The changes are container/layout and sparse coordinate encoding only, with dtype-preserving copied new values and `#old` values from the pinned base. No changed coordinate or replacement value is added, dropped or altered. The patch is a maintainer-produced canonical representation, not the author's published patch file. The original edit's credit, license obligations and prior changes remain attributable to its source.
