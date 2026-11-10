# Attribution, component terms and changes

Original study: **Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab**, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), published 2026-11-02. Credit and source identification do not imply endorsement. Study outcomes are not independently reproduced here and are not inherited by this transcode or any future derivative.

## Base weights

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), revision `2fc06364715b967f1860aea9cf38778875588b17`, is supplied under Apache-2.0, Copyright 2026 Alibaba Cloud. Retain [Apache-2.0-Qwen.txt](Apache-2.0-Qwen.txt) and applicable notices with base-derived material. The retained source qualification records found no separate NOTICE file in the pinned HF listings. The [patched reference](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/tree/d1671a178705bcab7644806d23c535517f35723b) supplied the same license text. Full checkpoints are not distributed in this index snapshot.

## Compact edit and lossless transcode

Original compact source: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit), revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, files `exp8_math/runs/p50c/edit/edit.npz` and `manifest.json`. The [source license allocation](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE) assigns logs, data and networks, including this edit, **CC BY 4.0**. Preserve the [allocation notice](SOURCE-LICENSE-lm_understand-logs.txt), author/article/source credit, applicable disclaimers and any prior source modification history. Terms: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en). No endorsement is implied, and third-party benchmark or dataset content is not included here.

Changes dated 2026-11-10: the original sparse NPZ representation is losslessly transcoded to complete selected-column safetensors slabs with explicit whole-base/source identities and versioned replacement semantics. The representation includes **884 unchanged base-identical addressed positions**, preserves all **130,188 actual changed coordinates and BF16 bit patterns**, and reproduces every final stored tensor. The exact payload is **264,800 bytes**, SHA-256 `527139e3ec7870416c6d8e22026d884273cd9221444041114bea44edabd9fedc`. Catalog records, byte evidence and explanatory documents are additional index material. The convention is agreed as **wsi-sparse-exact v1**; the byte-identical payload retains its original historical proposal wording in a descriptive metadata string. That string is preserved rather than re-encoding the file and does not change the versioned replacement operation.

## Scripts and index material

The publisher's source LICENSE allocates MIT terms to its scripts. **No publisher script implementation is copied, executed or redistributed** in this entry. The retained allocation notice includes those terms as source context; they do not relicense the compact edit or this payload under MIT. Applicable base-weight Apache duties remain alongside CC BY duties; changing representation does not remove either layer.

Rights in new index documentation and records remain a separate repository-owner decision. No repository LICENSE file is supplied. License labels and grants do not establish a verified chain of title.
