import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
// A layout engine is needed to measure scrollbars, so this test pins the CSS
// rule that removes them instead (see PostLayout.astro). Astro may inline the
// rule into the page or put it in a CSS file, so both are read.
const POST = 'dist/posts/2025/11/VAE-to-DDPMs/index.html';

const pageCss = (): string =>
  [
    readFileSync(POST, 'utf8'),
    ...readdirSync('dist/_astro')
      .filter((f) => f.endsWith('.css'))
      .map((f) => readFileSync(join('dist/_astro', f), 'utf8')),
  ].join('\n');

describe('display math', () => {
  it('hides the vertical overflow and pads the box so equations that fit show no scrollbar', () => {
    const rules = pageCss().match(/\.article-body \.katex-display\{[^}]*\}/g) ?? [];
    const fix = rules.find(
      (r) => r.includes('overflow:auto hidden') || r.includes('overflow-y:hidden'),
    );
    expect(fix).toBeDefined();
    expect(fix).toMatch(/padding:\.5em \.25em/);
  });
});
