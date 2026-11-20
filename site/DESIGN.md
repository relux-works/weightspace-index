# Design contract

The site uses the Relux product-family web design system. This file records how it is applied here and where it deliberately differs.

## Reader journey

The catalog serves someone who has a base model and wants to find, evaluate and run a compatible patch. The primary route is the directory, a model and a patch card.

1. **Directory.** The desktop directory gives model discovery the wider column and insets its single worked example from the top. Filtering by model, goal, patch name or author updates the list in place. Model rows show namespace and active patch count. The example shows matching base and patched grids from one sampled weight matrix and an attributed benchmark result. Prefer a Weightspace Index result even when its measured change is zero or negative. On narrow screens the example sits beneath the model rows, aligns to their gutter, and caps at 36rem. A Patch vs LoRA section follows the directory and crossfades between Patch and LoRA in the middle of the weight equation. Two faint dots indicate the current slide. Hovering over the diagram pauses it, and reduced motion shows both forms together. The introduction and footer link directly to it.
2. **Model.** Each patch row shows its goal, author, release, profile, download shortcuts and a link to the patch card. One representative result per source keeps the list scannable. Compare patches gives each benchmark one band. Inside it, the Reported group (patch authors' own figures) always comes first and the Measured group (index runs) second, each with its own unpatched base. A group reads like a grouped table: its source name, any item count it shares, and its metric when the two sources name the benchmark's metric differently (an author's "accuracy", bench's "final answers right"), sit in a narrow left column with a hairline running down beside its rows (no dot, no header row); on narrow screens the name sits above the rows over a hairline. Each group lists the unpatched base, each patch measured on it, then each measured stack of active patches (pairs before triples, in patch order). A patch or stack with no score there is left out; a run still being scored keeps a progress row. Each row prints its score and its change beside the bar; rows in one group measured against different bases each draw their own base bar and print "from" that base. Bars are colored by what was applied, never by direction: red for a patch, green for a stack; the change column says better or worse. Benchmark bands alternate a tint, with equal space above the first bar and below the last. Base files and run records are expandable.
3. **Patch.** The goal, credit and exact base revision sit beside the available downloads. The runtime tabs show launch commands, with one button to copy the entire recipe and a disclosure to inspect download and preparation steps. The LoRA button preserves the reader’s selected adapter runtime. Section links jump to measurements and release details. A line under the measurements names the measured stacks that include the patch and links to the model page. Detailed tensor maps, provenance, reproduction commands and version history live in disclosures.
4. **How it works** (`/how-it-works/`). An optional explainer, not tied to any one model, for readers who need the concept before a format decision. The opening and the two forms form one reading column, an article:
   - The opening: one outlined weight traced from its base value to its patched value.
   - The same edit in two forms, in sequence: a replacement stores the new value, and a LoRA stores two factors whose product is the change, with the arithmetic printed step by step. Each form is a section of the article. Bridge lines sit before the first form, between the two, and after the second.
   - "Why we ship both": the choice first, as two columns (LoRA adapter, direct replacement), each with one line of use. The LoRA column names the vLLM, llama.cpp, SGLang and PEFT docs and notes that the index's derived adapters are not yet verified in those runtimes. Then three drawn comparisons, one row each: speed (attached versus baked lanes), precision (the stored value versus the computed one), and size (shared-scale bars for two shapes, with the rank and dtype stated). No cards and no kickers; each drawing carries its point.
   - A file-size calculator for any matrix shape, with the featured patch's shipped sizes.
   - A closing action into the directory and a link to the featured patch's results. It names the featured result only; it makes no claim that every patch is benchmarked.

Header navigation is Models, How it works, Search. The full benchmark run list (`/compare/`) is linked from the footer.

When a measurement is missing or in progress, the page shows a short status with a progress bar. Long patch directories show eight entries at a time and search every entry.

## Visual system

Inter Tight sets headings, Inter sets UI copy and Roboto Mono sets code, numbers and tensor names only (not labels). The catalog keeps the Relux mark, red accent, theme tokens and dark footer.

- **Weights.** The featured patch samples a region with edited weights from the replacement file. The two grids use matching cells. Varied grays illustrate the base values and red intensity shows the magnitude of recorded edits. The benchmark footer uses horizontal bars on a shared scale and colors the patched score red. The non-endorsement note sits below the bars in smaller, muted text aligned to the right. A 1rem gap on each side of the dividers separates the title, matrices, and benchmark. The patch card has a more detailed edit map.
- **Evidence.** Label each result with its source, in words: "Reported" (the patch authors' own published figure) or "Measured" (a run of the index's own benchmark, filed with every request and reply, by the index team or any contributor), never a bare dot. The words say where a number comes from, not whom to trust: the index provides the receipts. The patch page shows one chart per benchmark with base, Reported and Measured rows, every score printed beside its bar, and a status that says only whether the measurement went the way of the reported figure (Confirmed, Not reproduced, Measured only, No gain found, Being measured, Reported only). A Reported row the authors never reported and a Measured row that was never run are left out; one still being scored reads "Score pending". A progress count names its unit, such as item–seed pairs. Every chart prints the values. Changes in percent read in percentage points, "pp". On a patch card, hovering a benchmark's chart floats a one-line legend above it, naming only the marks that chart uses; touch screens get one legend under the charts instead. The score plots use a shared axis within each unit and show both treatments at once.
- **Claims match the files.** The replacement is pitched as exact and free at run time, not as the smaller file. The size calculator compares sizes before compression and names one real release's download sizes next to the model.
- **Icons are drawn.** Arrows come from `components/Arrow.astro` (right, out, down), never from typed glyphs.
- **Matrix illustrations.** The featured patch, Patch vs LoRA slides and explainer article share a cell-size ceiling and square-to-gap ratio. The article uses the homepage comparison’s neutral surface, gray weights and red edits.
- **Dark ink surfaces** (featured patch and terminals) carry their own light focus ring and dark scrollbars.
- **Charts.** These are our own Astro components; there are no third-party chart or highlighting libraries. Shell highlighting is `lib/shell.ts`.
- **Numbers.** Every chart also prints its values as text.
- **Motion.** Directory evidence stays still while the reader compares it. The optional explainer and runtime tabs respect reduced motion.
- **Separate sources.** Reported and Measured results are never ranked together. A patch card or patch page may draw them as rows of one chart, each row labelled with its source and its change taken from that source's own base; the author's report is a solid bar with its change printed in the chart red, an even OKLCH mix of the link red and the Relux identity red (#ce090c light, #fc5850 dark), and the Measured result is a ghost of it, a faint fill inside a red outline (owner's decision). The patch page sets the charts in one column. Compare patches on the model page follows the same rule: a benchmark band holds a Reported group and a Measured group, each labelled with its source and changed from its own base. Elsewhere each block names its source.
- **Copy is a label, not a paragraph.** Readers skip prose. A drawing, a number or a two-word label carries the point; a sentence appears only where nothing else can say it, and never repeats what a drawing above it already shows.
- **Switches glide, timers stay hidden.** Auto-advancing steps carry no countdown bar. The chosen tab, step or switch is marked by one indicator (a thumb or a 2px line, `scripts/slide.ts`) that glides to the new choice, and the content fades in with a short rise. Reduced motion turns both off.
- **Without JavaScript.** All content stays readable: every runtime tab, every measured stack and the illustrated format guide are visible.
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
