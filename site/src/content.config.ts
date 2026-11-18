// Patch cards, read straight from the index's models/ folder at build time.
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { DEMO, PREFIX } from "./lib/config.ts";

const cards = defineCollection({
  loader: glob({
    pattern: DEMO ? ["models/*/*/*/README.md", "docs/example/models/*/*/*/README.md"] : [`${PREFIX}models/*/*/*/README.md`],
    base: new URL("../../", import.meta.url),
    // Keep the exact case of the base and slug: "<ns>/<model>/<slug>".
    generateId: ({ entry }) => entry.replace(/^(?:docs\/example\/)?models\//, "").replace(/\/README\.md$/, ""),
  }),
});

export const collections = { cards };
