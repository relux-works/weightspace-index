# thinkingmachines/Inkling-2-Mini

Generated from the files in this folder by `weightspace generate`; do not edit.

## Profiles

| Profile | Source | Files |
|---|---|---|
| `nvfp4` | [thinkingmachines/Inkling-2-Mini@c9d0283](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195) | 2 |

## Patches

| Patch | Kind | Latest | Summary |
|---|---|---|---|
| [ex0bit-inkblot-2-mini](ex0bit-inkblot-2-mini/) | replacement | v1 | Edits 327 BF16/FP32 loop-control and routing weights in Inkling-2-Mini to stabilize multi-pass looping and improve math and code reasoning while leaving all 24.4B quantized experts untouched. |

## Results

Latest active version of each patch, against the baseline run on the same profile.

| Patch | Profile | Metric | Baseline | Patched |
|---|---|---|---|---|
| ex0bit-inkblot-2-mini v1 | `nvfp4` | not measured | | | |
