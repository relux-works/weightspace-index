# Design contract

The site uses the Relux product-family web design system. This file records how it is applied here and where it deliberately differs.

## Reader journey

The index exists so readers can find, pick and download patches. Most readers see three pages and nothing deeper: home, a model, a patch. Each page puts the next action first and keeps the details in folds.

1. **Home.** The headline is “Patch your model.” Next to it, a live example toggles a featured patch between base and patched: its edited columns drawn on the weight matrices (red, brighter for larger changes), its scores and its file size. The model directory follows as a plain list (model, patches, best result). A short visual section shows base + patch = a better model as heatmaps, then the two formats. A closing row says who measured each result.
2. **Model.** The h1 is the model address. One meta line carries patch count, formats, tested combinations and a link to the base weights. Then:
   - **Patches.** One row per patch: name, author, goal, its top gains with their source, and the two download buttons.
   - **Compare patches** (two or more patches). One table per source: benchmark, patch, base, patched, change.
   - **Patch combinations** (two or more patches). A matrix of patch pairs; a cell opens that study's bars.
3. **Patch.** The page opens with the goal, the credit line and two large download buttons: the LoRA adapter first, then the direct replacement. These are the reason the index exists and carry the most visual weight on the page. Each button also selects its setup tab. After that:
   - **Results.** A self-reported list and an independently measured list, each with its own label; runs in progress show a progress bar.
   - **Use it.** One tab per runtime. Each tab is a single highlighted shell script that starts with a `curl` download, so the reader copies once and runs it. The weightspace tab bakes the replacement.
   - **What changes.** For each edited matrix: where the edited columns sit, and a heatmap of sampled rows from the real old and new values. Hovering a cell reads it out.
   - **Combinations**, then release details, hashes and version history in folds.
4. **How it works** (`/how-it-works/`). An optional explainer that is not tied to any one model:
   - A three-step drawing (weights, W + ΔW = W′, then W + B × A = the same W′) that cycles on its own until the reader picks a step.
   - A “which file do I need” picker that cycles the same way.
   - A file-size calculator for any matrix shape.
   - A link to the most-benchmarked patch's results.

   Auto-cycling pauses on hover, on focus and off screen, stops on the first click, and never starts under reduced motion. All steps share one grid cell, so the height stays stable.

Header navigation is Models, How it works, Search. The full benchmark run list (`/compare/`) is linked from the footer.

When a measurement is missing or in progress, the page shows a short status with a progress bar. Long patch directories show eight entries at a time and search every entry.

## Visual system

Inter Tight sets headings, Inter sets UI copy and Roboto Mono sets code, numbers and tensor names only (not labels). The catalog keeps the Relux mark, red accent, theme tokens and dark footer.

- **Weights as heatmaps.** Blue cells are weights; red cells are changes, with opacity linear in the size of the change. Unchanged cells inside an edited column stay empty. The home page draws weights in a light grey so only the red carries weight (the live example on ink, and the outlined matrices of "A patch is a tiny, exact change"). The heading sits beside one centred equation (stacked above it on narrow screens): your model + a patch = a better model, with the patch drawn as a replacement and as LoRA factors that crossfade in one slot on a seven-second loop, captioned only “Patch” and “LoRA”, with no buttons, “or” or divider; under reduced motion the two sit stacked and still. The matrices are square (12 × 12, three edited columns, a rank-3 LoRA) and every one is sized from a single cell size, so a cell is the same size in W, ΔW, B, A and W′. Labels hang below their matrix, so + and = sit on the matrices' midline. Only the crossfade moves. The explainer's figures and the picker's answer use dark ink surfaces in both themes.
- **No chips or pills.** Source labels are text with a marker (hollow ring for self-reported, filled dot for independently measured). Deltas and verdicts are colored text. Small labels use sentence case, never uppercase mono. Headings carry no eyebrow above them.
- **Every number names its source.** Wherever a result appears (directory row, patch card, home equation, live example, combination study) its marker sits next to it. A measured result with no gain shows as ±0, never as "not measured".
- **Red means change.** Red marks edited cells and the brand accent only. Patched and combined values are ink, single patches and baselines grey, and green or red appear only as the text of a delta or verdict.
- **Claims match the files.** The replacement is pitched as exact and free at run time, not as the smaller file. The size calculator compares sizes before compression and names one real release's download sizes next to the model.
- **Icons are drawn.** Arrows come from `components/Arrow.astro` (right, out, down), never from typed glyphs.
- **Dark ink surfaces** (live example, explainer figures, picker answer, terminals) carry their own light focus ring and dark scrollbars.
- **Charts.** These are our own Astro components; there are no third-party chart or highlighting libraries. Shell highlighting is `lib/shell.ts`.
- **Numbers.** Every chart also prints its values as text.
- **Motion.** Only `transform` and `opacity` animate. The live example fades two stacked layers rather than thousands of cells; it pauses under the pointer or keyboard focus, stops once the reader picks a state, and its moving numbers are hidden from assistive tech, which reads a static base/patched pair. Animation respects `prefers-reduced-motion`.
- **Separate sources.** Self-reported and independently measured results are never ranked together or placed in one chart. Each block names its source.
- **Copy is a label, not a paragraph.** Readers skip prose. A drawing, a number or a two-word label carries the point; a sentence appears only where nothing else can say it, and never repeats what a drawing above it already shows.
- **Switches glide, timers stay hidden.** Auto-advancing steps carry no countdown bar. The chosen tab, step or switch is marked by one indicator (a thumb or a 2px line, `scripts/slide.ts`) that glides to the new choice, and the content fades in with a short rise. Reduced motion turns both off.
- **Without JavaScript.** All content stays readable: every runtime tab, every combination study, every explainer step and every picker answer is visible.
- **Illustrations.** The explainer's matrices and example values are illustrative. The patch page's heatmaps read the release's own replacement file.

- The footer carries the trademark notice: GitHub and the GitHub logo are trademarks of GitHub, Inc.; model names belong to their respective owners; their use implies no endorsement.

## Intentional deviations

| Family default | Here | Reason |
|---|---|---|
| h1 3.5rem, h2 2.75rem | h1 3.2rem on catalog detail pages, h2 1.75rem, home statement 4.4rem | Pages are dense reference material with many sections; large headings would push data below the fold. |
| Section padding 6rem (4rem mobile) | 3rem (2rem mobile) | Same reason: operational density. |
| Mobile navigation drawer | Navigation on a second header row | There are only three links; a drawer would hide them behind an extra tap. |
| RTL and locale coverage | English only, LTR | The index and its records are English-only. |
| Footer uses the theme's subtle surface | In the light theme the footer is an ink surface (`#181a18`) with the dark-theme text, link and focus roles and the dark lockup; forced colors keep the system palette | Owner's decision: the footer closes every page as a dark band. Its colors are the family's own dark-theme roles, scoped in `tokens.css`. |
| Product keeps its own mark, separate from the Relux symbol; the symbol pairs only with the full name nearby | Header pairs the Relux symbol with the live-type product name "Weightspace Index" | Owner's decision: the index reads as a Relux product. The symbol artwork is unchanged, and the outlined Relux wordmark is not altered or re-typed. |

## Validation

- Widths: 390, 768, 1440 and 1920 CSS px.
- Themes: light and dark.
- Routes: all routes in the real build and in the example build.
- Checks:
  - no horizontal overflow and a stable header height;
  - anchor targets clear the sticky header;
  - copy controls don't resize.
- Not yet run: the automated axe-core gate, a screen-reader pass, and manual checks of increased contrast and forced colors. They are required before claiming WCAG 2.2 AA.
