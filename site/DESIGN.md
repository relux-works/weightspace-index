# Design contract

The site uses the Relux product-family web design system. This file records how it is applied here and where it deliberately differs.

## Product

- **Promise:** an open index of weight patches for open models. For each patch it shows what changes, the exact base files it targets, where to download it, and what was measured.
- **Archetype:** documentation and operational tool. The first screen is the working surface: product name, a one-line statement, search and the index counts. There is no promotional hero.
- **Audience:** people who run open models and want to find, verify, apply or benchmark a patch.
- **Proof:** every value comes from a record at the pinned index commit. File rows show size and SHA-256. Commands are filled in from records and link the document that defines them.

## Family invariants

- **Tokens:** `src/styles/tokens.css` is the family token contract, copied unchanged except for the font stacks. It covers light, dark, increased contrast (light and dark), reduced transparency and forced colors. Components use semantic roles only, never literal colors.
- **Type:**
  - Inter Tight for display, Inter for UI and Roboto Mono for technical text, all self-hosted from Fontsource.
  - `letter-spacing: 0` everywhere, on a fixed responsive scale with no viewport-tied sizes.
- **Geometry:**
  - Radii are 4px on controls and 8px on surfaces.
  - Layout is full-width bands on a constrained grid (`--layout-max`, 73.75rem).
  - Section headings use the 1.25fr / 0.75fr split.
  - Cards are used only for repeated framed items (benchmark groups, comparisons, assessments, sessions), and cards are never nested. Blocks that contain framed commands are separated by dividers instead.
- **Header:**
  - Three stable zones at 4.25rem: the product name, centered navigation, and the GitHub link.
  - The GitHub link is the 20px GitHub mark plus "View on GitHub", styled like the nav links. The label describes the action rather than standing alone as "GitHub", so it doesn't read as a GitHub-branded or endorsed item. It opens the repository tree at the pinned commit, so it says where the data lives. On phones only the mark shows. It opens in a new tab. Its accessible name is "View on GitHub at commit … (opens in a new tab)" and its tooltip gives the full commit. The commit itself is shown in the home eyebrow and in the footer of every page.
  - At tablet width the side zones shrink to their content.
  - On phones the navigation moves to a second row.
- **States:**
  - Badges use a shape as well as a color, so state is never conveyed by hue alone: circle for ok and info, triangle for warning, square for error, ring for neutral.
  - The copy button keeps its size when it changes to "Copied".
- **Motion:** press feedback is `scale(.98)`, and color transitions are short. Reduced-motion users get no transitions.

## Expression slots

- **Accent:** the Relux red ramp from the baseline palette. It marks links, the active navigation item, step numbers and patched bars in charts. Base bars are hatched, so they don't depend on hue either.
- **Neutral tint:** the baseline green-grey neutrals.
- **Density:** operational. See the deviations below.
- **Media:** none. The data is the content.

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
| h1 3.5rem, h2 2.75rem | h1 2.75rem, h2 1.75rem (home statement 4.5rem) | Pages are dense reference material with many sections; large headings would push data below the fold. |
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
