import { readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
const URL = '/posts/2024/05/Tensor-Program-2/';
const page = `dist${URL}index.html`;
const reference = JSON.parse(readFileSync('tests/fixtures/jekyll-reference.json', 'utf8'));

describe(`citations on ${URL}`, () => {
  const html = readFileSync(page, 'utf8');
  const article = parse(html).querySelector('.article-body');

  it('renders the article body', () => {
    expect(article).not.toBeNull();
  });

  it('cites the same distinct keys as Jekyll', () => {
    const targets = new Set(
      article!.querySelectorAll('a[href^="#bib-"]').map((a) => a.getAttribute('href')),
    );
    expect(targets.size).toBe(reference.posts[URL].citations);
  });

  it('lists the same number of references as Jekyll', () => {
    const entries = article!.querySelectorAll('#refs .csl-entry');
    expect(entries.length).toBe(reference.posts[URL].references);
  });

  it('renders math with KaTeX', () => {
    expect(article!.querySelectorAll('.katex').length).toBeGreaterThan(0);
  });
});
