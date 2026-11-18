// Patch cards, read straight from the index's models/ folder at build time.
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { PREFIX } from "./lib/config.ts";

const cards = defineCollection({
  loader: glob({
    pattern: "*/*/*/README.md",
    base: new URL(`../../${PREFIX}models/`, import.meta.url),
    // Keep the exact case of the base and slug: "<ns>/<model>/<slug>".
    generateId: ({ entry }) => entry.replace(/\/README\.md$/, ""),
  }),
});

export const collections = { cards };
