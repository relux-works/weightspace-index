import { defineConfig } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import { DEMO, EXAMPLE, SITE_BASE } from "./src/lib/config.ts";
import { cardHast, cardMdast } from "./src/lib/card-plugin.ts";

export default defineConfig({
  base: SITE_BASE,
  outDir: EXAMPLE ? "./dist-example" : DEMO ? "./dist-demo" : "./dist",
  // Pages build to dir/index.html and every internal link ends in "/". "ignore"
  // keeps that output while dev and preview also serve /page without the slash,
  // as static hosts do by redirecting, instead of answering 404.
  trailingSlash: "ignore",
  devToolbar: { enabled: false },
  markdown: {
    // Cards are shown as authored: no typographic rewriting, no highlighter markup.
    processor: satteri({
      features: { smartPunctuation: false },
      mdastPlugins: [cardMdast],
      hastPlugins: [cardHast],
    }),
    syntaxHighlight: false,
  },
});
