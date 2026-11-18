// Build settings shared by astro.config.ts and the pages. All come from the
// environment so the same code builds the real index and the example fixture.

/** Which collection to read: "" for the index, "docs/example" for the synthetic fixture. */
export const COLLECTION = (process.env.WS_INDEX_ROOT ?? "").replace(/^\/+|\/+$/g, "");
if (COLLECTION !== "" && COLLECTION !== "docs/example") {
  throw new Error(`[weightspace site] WS_INDEX_ROOT must be empty (the index) or docs/example (the synthetic fixture), got ${COLLECTION}`);
}
export const EXAMPLE = COLLECTION === "docs/example";
export const DEMO = !EXAMPLE && process.env.WS_SITE_DEMO === "1";
/** Repository path prefix of the collection ("" or "docs/example/"). */
export const PREFIX = COLLECTION ? `${COLLECTION}/` : "";

/** GitHub repository that download links and source views point at. */
export const GITHUB_REPO = process.env.WS_GITHUB_REPO ?? "relux-works/weightspace-index";

/** URL path the site is served under, e.g. "/" or "/weightspace-index/". */
export const SITE_BASE = `/${(process.env.WS_SITE_BASE ?? "").replace(/^\/+|\/+$/g, "")}/`.replace(/\/+/g, "/");

export function href(p = ""): string {
  return SITE_BASE + p.replace(/^\/+/, "");
}

export function seg(p: string): string {
  return p.split("/").map(encodeURIComponent).join("/");
}

/** Paths whose uncommitted changes would make download links point at other bytes. */
export const PIN_PATHS = EXAMPLE ? ["docs/example"] : ["models", "lookup", "bench", "schema", ...(DEMO ? ["docs/example"] : [])];
