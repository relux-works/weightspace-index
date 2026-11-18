import { defineConfig } from "astro/config";
import { satteri } from "@astrojs/markdown-satteri";
import { DEMO, EXAMPLE, SITE_BASE } from "./src/lib/config.ts";
import { cardHast, cardMdast } from "./src/lib/card-plugin.ts";

export default defineConfig({
  base: SITE_BASE,
  outDir: EXAMPLE ? "./dist-example" : DEMO ? "./dist-demo" : "./dist",
  trailingSlash: "always",
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
