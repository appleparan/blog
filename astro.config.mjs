import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import tailwindcss from '@tailwindcss/vite';
import viteCompression from 'vite-plugin-compression';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeCitation from 'rehype-citation';
import { remarkQuoteCite } from './src/lib/remark-quote-cite';
import { rehypeReferencesHeading } from './src/lib/rehype-references-heading';

export default defineConfig({
  compressHTML: true,
  integrations: [mdx()],
  vite: {
    plugins: [tailwindcss(), viteCompression({ algorithm: 'gzip', threshold: 1024 })],
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
        [rehypeCitation, { path: 'src/content/bibliography', csl: 'ieee.csl', linkCitations: true }],
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
