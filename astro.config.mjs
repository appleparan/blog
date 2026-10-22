import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import tailwindcss from '@tailwindcss/vite';
import viteCompression from 'vite-plugin-compression';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  compressHTML: true,
  integrations: [mdx()],
  vite: {
    plugins: [tailwindcss(), viteCompression({ algorithm: 'gzip', threshold: 1024 })],
  },
  markdown: {
    // Astro 7 defaults to the Satteri processor, which does not run remark/rehype
    // plugins. remark-math and rehype-katex need the unified processor.
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
    shikiConfig: {
      theme: 'one-dark-pro',
    },
  },
});
