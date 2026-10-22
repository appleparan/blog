import { copyFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { AstroIntegration } from "astro";

/**
 * Copies `sitemap-index.xml` to `sitemap.xml`, the URL the Jekyll site used and robots.txt
 * names. The index lists the absolute `sitemap-0.xml` URL, so the copy stays valid.
 */
export function sitemapAlias(): AstroIntegration {
  return {
    name: "sitemap-alias",
    hooks: {
      "astro:build:done": ({ dir }) => {
        const out = (name: string) => fileURLToPath(new URL(name, dir));
        copyFileSync(out("sitemap-index.xml"), out("sitemap.xml"));
      },
    },
  };
}
