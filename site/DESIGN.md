# Design contract

The site uses the Relux product-family web design system. This file records how it is applied here and where it deliberately differs.

## Reader journey

The catalog serves someone who has a base model and wants to find, evaluate and run a compatible patch. The primary route is the directory, a model and a patch card.

1. **Directory.** The desktop directory gives model discovery the wider column and insets its single worked example from the top. Filtering by model, goal, patch name or author updates the list in place. Model rows show namespace and active patch count. The example shows matching base and patched grids from one sampled weight matrix and an attributed benchmark result. Prefer a team or contributor result even when its measured change is zero or negative. On narrow screens the example sits beneath the model rows, aligns to their gutter, and caps at 36rem. A Patch vs LoRA section follows the directory and crossfades between Patch and LoRA in the middle of the weight equation. Two faint dots indicate the current slide. Hovering over the diagram pauses it, and reduced motion shows both forms together. The introduction and footer link directly to it.
2. **Model.** Each patch row shows its goal, author, release, profile, download shortcuts and a link to the patch card. One representative result per performer keeps the list scannable. Larger catalogs can compare the full results by source. The combinations section shows which current pairs have been tested. With three or more active patches, a matrix lets the reader select a measured pair. A two-patch model opens directly on its studies. Each study identifies its exact releases, individual scores and joint score. Historical studies and studies of three or more patches remain available below the matrix. Base files and run records are expandable.
3. **Patch.** The goal, credit and exact base revision sit beside the available downloads. The runtime tabs show launch commands, with one button to copy the entire recipe and a disclosure to inspect download and preparation steps. The LoRA button preserves the reader’s selected adapter runtime. Section links jump to measurements, combinations and release details. Detailed tensor maps, provenance, reproduction commands and version history live in disclosures.
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

- **Weights.** The featured patch samples a region with edited weights from the replacement file. The two grids use matching cells. Varied grays illustrate the base values and red intensity shows the magnitude of recorded edits. The benchmark footer uses horizontal bars on a shared scale and colors the patched score red. The non-endorsement note sits below the bars in smaller, muted text aligned to the right. A 1rem gap on each side of the dividers separates the title, matrices, and benchmark. The patch card has a more detailed edit map.
- **Evidence.** Label each result with its performer. Author reports, index team measurements and contributor measurements each have their own group. A progress count names its unit, such as item–seed pairs. Every chart prints the values. The score plots use a shared axis within each unit and show both treatments at once.
- **Claims match the files.** The replacement is pitched as exact and free at run time, not as the smaller file. The size calculator compares sizes before compression and names one real release's download sizes next to the model.
- **Icons are drawn.** Arrows come from `components/Arrow.astro` (right, out, down), never from typed glyphs.
- **Dark ink surfaces** (featured patch, explainer figures, picker answer, terminals) carry their own light focus ring and dark scrollbars.
- **Charts.** These are our own Astro components; there are no third-party chart or highlighting libraries. Shell highlighting is `lib/shell.ts`.
- **Numbers.** Every chart also prints its values as text.
- **Motion.** Directory evidence stays still while the reader compares it. The optional explainer and runtime tabs respect reduced motion.
- **Separate sources.** Self-reported and independently measured results are never ranked together or placed in one chart. Each block names its source.
- **Copy is a label, not a paragraph.** Readers skip prose. A drawing, a number or a two-word label carries the point; a sentence appears only where nothing else can say it, and never repeats what a drawing above it already shows.
- **Switches glide, timers stay hidden.** Auto-advancing steps carry no countdown bar. The chosen tab, step or switch is marked by one indicator (a thumb or a 2px line, `scripts/slide.ts`) that glides to the new choice, and the content fades in with a short rise. Reduced motion turns both off.
- **Without JavaScript.** All content stays readable: every runtime tab, every combination study, every explainer step and every picker answer is visible.
- **Illustrations.** The explainer's matrices and example values are illustrative. The patch page's heatmaps read the release's own replacement file.

- The footer carries the trademark notice: GitHub and the GitHub logo are trademarks of GitHub, Inc.; model names belong to their respective owners; their use implies no endorsement.

## Intentional deviations

| Family default | Here | Reason |
|---|---|---|
| h1 3.5rem, h2 2.75rem | Responsive h1 up to 4.75rem on the directory, 4.3rem on patch cards, h2 1.75rem | Pages are dense reference material with many sections; large headings would push data below the fold. |
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
