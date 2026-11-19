# Weightspace Index site

The static Astro directory filters models in place by model, patch, purpose or author. The featured patch uses a before-and-after sample from its weight matrix to show how a replacement changes weights. The root page shows when to choose a direct replacement or a LoRA adapter, with a crossfade between the two representations of an illustrative edit. The model page presents each patch’s goal, downloads and a benchmark snapshot grouped by performer. The stable patch URL holds the latest active release: its downloads (direct replacement and LoRA), results, runtime commands, edit map and combination studies. Version URLs stay available for reproducibility. `/how-it-works/` explains patches and LoRAs visually. See `DESIGN.md` for the reader journey.

## Run locally

Use Node 22.23.4 or newer and Rust for the offline index check.

```sh
cd site
npm ci
npm run dev            # production catalog
npm run dev:demo       # real catalog plus the labeled synthetic model
npm test
npm run build          # dist/
npm run build:demo     # dist-demo/
npm run build:example  # synthetic fixture alone in dist-example/
npm run preview        # serve dist/
```

The default production build excludes the demo model, its files and its combination study. Demo and example pages carry a robots noindex directive. The demo model has a visible label and a synthetic-data banner on its pages.

## Inputs and outputs

- The reader checks the index contract, local digests, profiles and run references. Builds run `weightspace check` against a Git archive of HEAD. A production build stops when the index inputs differ from HEAD.
- In-repository downloads are served from the site under their SHA-256 and filename. External artifacts use their pinned locators. Each supported canonical LoRA can produce a PEFT bundle with the original factor bytes, runtime configuration, notices and provenance. A verified architecture-specific GGUF export can travel in the same archive.
- Each published author study in `src/data/reports.ts` names its source and applies to a specific release artifact and base revision. When the homepage features a result from an original study, the benchmark links to the article in a new tab and states that the listing is independent of the study authors. The rest of the card opens the patch page. Independent measurements come from run records and keep their performer, profile, settings and completion state.
- Optional joint studies are read from `bench/combinations/<namespace>/<model>/`. The synthetic example lives below `docs/example/bench/combinations/`. See the [benchmark notes](../docs/benchmarking.md#combination-studies).
- Each runtime tab shows the launch command. Its copy button includes the full download, checksum, preparation and launch recipe. Without JavaScript, readers can download the file and expand the same preparation steps. Directory search, paging, runtime tabs and copy buttons use local JavaScript. Navigation, the complete directory, download links, evidence and every recipe work as static HTML. Fonts and visualizations are served locally.

## Build settings

| Variable | Effect |
|---|---|
| `WS_SITE_BASE` | URL prefix when serving below a subdirectory. |
| `WS_GITHUB_REPO` | Owner/repository for pinned source links. Defaults to `relux-works/weightspace-index`. |
| `WS_INDEX_ROOT=docs/example` | Read the synthetic fixture alone. |
| `WS_SITE_DEMO=1` | Include the fixture beside the real catalog. |
| `WS_SKIP_INDEX_CHECK=1` | Skip the Rust check during local iteration. |
| `WS_ALLOW_DIRTY=1` | Preview uncommitted data with a visible development banner. |

See [DESIGN.md](DESIGN.md) for the layout and evidence presentation.
