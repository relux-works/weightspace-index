# Site

A static site over the index, built with Astro. It only reads `models/`, `lookup/`, `bench/`, `schema/` and `docs/`; it never writes them.

## Build

Needs Node 22.23.4 or newer, and Rust (for `weightspace check`).

```bash
cd site
npm ci
npm run build          # real index → dist/
npm run build:example  # docs/example fixture → dist-example/, marked as synthetic on every page
npm run preview        # serves the last build
npm test
```

`build` and `dev` first run `weightspace check` on a `git archive` snapshot of HEAD, then Astro. The build stops when:

- the check fails;
- a record the site reads is missing, is not valid JSON, or has a `contract_version` other than 1;
- a recorded size or SHA-256 does not match an in-repository file;
- an index file under the pinned paths differs from HEAD (non-dev builds only).

Every page, value and download link comes from the HEAD commit, shown in the header.

## Settings

| Variable | Effect |
|---|---|
| `WS_INDEX_ROOT` | `docs/example` builds the example fixture; empty (default) builds the index. |
| `WS_SITE_BASE` | URL base path when the site is not served at `/`. |
| `WS_GITHUB_REPO` | `owner/name` for download links (default `relux-works/weightspace-index`). |
| `WS_SKIP_INDEX_CHECK=1` | Skips `weightspace check` (local iteration only). |
| `WS_ALLOW_DIRTY=1` | Builds with uncommitted index changes; a banner says links point at the commit. |

## Runtime

No CDNs, analytics or third-party scripts. The pages use two small inline scripts: copy buttons, and search over `search.json`, which is generated at build time.
Fonts are self-hosted from the Fontsource packages and served with the site.

## Dependencies

Exact versions are pinned in `package.json` and `package-lock.json`.

| Package | Why |
|---|---|
| `astro` | Static site generator. |
| `@astrojs/markdown-satteri`, `satteri` | Astro's Markdown processor for patch cards. |
| `@fontsource-variable/inter` | UI typeface (Relux family type roles), self-hosted. |
| `@fontsource-variable/inter-tight` | Display typeface for headings, self-hosted. |
| `@fontsource/roboto-mono` | Technical typeface for digests, commands and labels, self-hosted. |

## Design

The site follows the Relux product-family web design system. The design contract, token source and intentional deviations are recorded in [DESIGN.md](DESIGN.md).
