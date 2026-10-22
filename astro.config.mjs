import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeCitation from 'rehype-citation';
import { sitemapAlias } from './src/integrations/sitemap-alias';
import { swVersion } from './src/integrations/sw-version';
import { remarkQuoteCite } from './src/lib/remark-quote-cite';
import { rehypeReferencesHeading } from './src/lib/rehype-references-heading';

export default defineConfig({
  compressHTML: true,
  site: 'https://blog.liam.kim',
  integrations: [
    mdx(),
    sitemap({ filter: (page) => !/\/404(\.html|\/)?$/.test(page) }),
    sitemapAlias(),
    swVersion(),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    // Astro 7 defaults to the Satteri processor, which does not run remark/rehype
    // plugins. remark-math and rehype-katex need the unified processor.
    processor: unified({
      remarkPlugins: [remarkMath, remarkQuoteCite],
      rehypePlugins: [
        rehypeKatex,
        // The .bib file name comes from each post's `bibliography` front matter,
        // which rehype-citation reads from `file.data.astro.frontmatter`.
        [
          rehypeCitation,
          { path: 'src/content/bibliography', csl: 'ieee.csl', linkCitations: true },
        ],
        rehypeReferencesHeading,
      ],
    }),
    // Keep ```mermaid blocks as raw <pre><code class="language-mermaid"> for Mermaid.astro.
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid', 'math'] },
    shikiConfig: {
      theme: 'one-dark-pro',
    },
  },
});
