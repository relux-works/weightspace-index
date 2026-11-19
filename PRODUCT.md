# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Practitioners** who run or fine-tune open models and must decide how to ship a weight edit: a replacement checkpoint or a LoRA adapter for their runtime.
- **Newcomers** who know LoRA by name but have not edited weights directly. They need the concept before any format decision.
- The `/how-it-works/` page serves both: a plain concept first, then the format decision for practitioners.

## Product Purpose

The Weightspace Index publishes weight patches for open models: what each patch changes, which exact model files it was made for, where to download it, and how it measured. `/how-it-works/` explains the underlying mechanism. Success for the page means a reader can say what a patch changes in a weight, and why the index ships each patch in two forms.

## Positioning

Every patch in the index ships as both a direct replacement (exact values written into a copy of the model) and a LoRA adapter (two low-rank factors loaded by a runtime). The two forms are the same change expressed two ways, and the page shows that with one outlined weight traced through both.

## Operating Context

- Patches are downloaded from Hugging Face at a pinned revision or from this repository at a pinned commit, then checked by SHA-256 and byte size.
- A replacement is applied once with the `weightspace` command-line tool to a copy of an unquantized model.
- A LoRA is loaded by the runtime: PEFT for Transformers, or GGUF for llama.cpp.

## Capabilities and Constraints

- A replacement works only on the target profile, which is unquantized. A LoRA works on any profile the runtime can attach it to, including GGUF quants.
- A replacement is exact. It is not automatically the smaller file. File size depends on the matrix shape, how many columns change, and how many values change within each column.
- LoRA runtime loading for derived adapters is unqualified except where documented. Study results do not transfer from one treatment to the other.
- Size figures are before compression.
- The illustrated matrices and example values are illustrative, not a real release.

## Brand Commitments

- The Relux Works product-family web design system: the Relux symbol and lockup, red accent, theme tokens, Inter Tight for headings, Inter for UI copy, Roboto Mono for code and numbers.
- English only, left to right.
- The owner has asked that this page follow the conventions of the root `/` page.

## Evidence on Hand

- `docs/patch-formats.md`: the replacement and LoRA forms, sparse replacement encoding, and derived-file rules.
- `site/src/data/reports.ts` and the index records: the most-benchmarked patch, used for the shipped-size line. It is read from the index, not hard-coded.
- `site/src/pages/how-it-works.astro`: the traced-weight example and its numbers.

No testimonials, customers, or benchmark claims beyond the index records may be added.

## Product Principles

1. Show the change before explaining it: one traced weight carries the point.
2. Claims match the files. Exact and free at run time, not smaller by default.
3. Two forms, one change. The reader should see the replacement and the adapter describe the same edit.
4. Text supports the drawing. A sentence appears only where nothing else can say the point.
5. Every number printed in a figure is also printed as text.

## Accessibility & Inclusion

- All content is readable without JavaScript. Motion respects reduced-motion preferences.
- The site's WCAG 2.2 AA claim is not yet validated (see `site/DESIGN.md`). Future work must not claim it before the checks there are run.
