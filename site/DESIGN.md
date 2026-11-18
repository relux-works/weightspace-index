# Design contract

The site uses the Relux product-family web design system. This file records how it is applied here and where it deliberately differs.

## Reader journey

1. **Directory.** Choose a model or filter by model, patch, author or format. Each tile lists releases by name and the available formats. A graph of actual edited input columns illustrates one downloadable patch. On phones it follows the directory.
2. **Model.** Scan the purpose, author, available formats and evidence status of each patch. Comparable completed runs share charts only within a profile, protocol, dataset, settings, metric, unit and performer category. A joint study displays the measured base, each patch alone and the combination.
3. **Patch.** Start with a short purpose and the active release’s downloads. Author findings and independently filed benchmarks have distinct bylines. Copy a recipe for the chosen runtime. Source cards, file hashes, notices and historical versions are in disclosures.

Models with few records use the same typography and download hierarchy as dense entries. One available patch stands on its own. Missing measurements use a short status. Long patch directories show eight entries at a time and search every entry. Multiple report groups, historical runs and setup details expand in place.

## Visual system

Inter Tight sets headings, Inter sets UI copy and Roboto Mono sets technical labels. The catalog retains the Relux mark, red accent, theme tokens and dark footer. The footprint and shell examples use dark ink surfaces in both themes. Model cards and download choices have one link across the full hit area. Numeric labels accompany charts and coverage bars. All runtime options are visible with JavaScript disabled, and enhanced tabs support arrow keys, Home and End.

## Relux Works relationship

- **Header:** the product mark is the unchanged Relux symbol with "Weightspace Index" beside it. The name is set in Inter Bold at the lockup's proportions (cap height 40.7% of the symbol box, starting 7.7% after it). The link's accessible name is the product name, and the symbol is hidden from assistive technology.
- **Footer:** Relux Works appears as the operator with the full horizontal lockup.
- The footer shows the horizontal lockup at 168px, linked to the relux-works GitHub organization. The link's accessible name is "Relux Works" and the mark is LTR-isolated.
- The lockup has light, dark and one-color variants. Under forced colors the one-color variant renders with the `CanvasText` fill.
- The artwork in `src/assets/brand/` and the favicons in `public/` are the approved SVGs, copied unchanged.
- The favicons are the Relux symbol, in light and dark variants that follow the color scheme.
- Source of the artwork and tokens: the relux-product-web-design guidance (MIT, Relux Works LLC).

## Third-party artwork

- **GitHub mark** (`src/assets/octicons/mark-github-16.svg`): Primer Octicons `mark-github` 16px, copied unchanged from primer/octicons at commit `68ec2de8d0ddfcbb7497da01d0b7d98eb5593020`. MIT, GitHub Inc.; the license is next to the file. It is shown in one color (`currentColor`), unaltered and only as a link to the index repository, in line with GitHub's logo guidelines. It implies no endorsement.
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
