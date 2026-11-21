// Patch cards and benchmark notes are read from the repository at build time.
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { DEMO, PREFIX } from "./lib/config.ts";

const roots = DEMO ? ["", "docs/example/"] : [PREFIX];
const repository = new URL("../../", import.meta.url);

const cards = defineCollection({
  loader: glob({
    pattern: roots.map(root => `${root}models/*/*/*/README.md`),
    base: repository,
    // Render with the current link rules and the build's pinned commit.
    deferRender: true,
    // Keep the exact case of the base and slug: "<ns>/<model>/<slug>".
    generateId: ({ entry }) => entry.replace(/^(?:docs\/example\/)?models\//, "").replace(/\/README\.md$/, ""),
  }),
});

const benchmarkNotes = defineCollection({
  loader: glob({
    pattern: roots.map(root => `${root}bench/notes/*/*/*/v[0-9]*.md`),
    base: repository,
    deferRender: true,
    generateId: ({ entry }) => entry.replace(/^(?:docs\/example\/)?bench\/notes\//, "").replace(/\.md$/, ""),
  }),
});

export const collections = { cards, benchmarkNotes };
