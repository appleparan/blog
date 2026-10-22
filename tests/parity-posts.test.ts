import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
interface Counts {
  images: number;
  math: number;
  citations: number;
  references: number;
}

const reference = JSON.parse(readFileSync('tests/fixtures/jekyll-reference.json', 'utf8')) as {
  posts: Record<string, Counts>;
};

// Posts whose counts differ from Jekyll on purpose. Each entry was checked on the page.
const KNOWN_DIFFERENCES: Record<string, { counts: Partial<Counts>; reason: string }> = {
  '/posts/2021/08/Topological-Overlap-Matrix/': {
    counts: { math: 37 },
    reason: 'the source typo `\\(\\)` gave Jekyll an empty math span; the MDX drops it',
  },
};

function builtCounts(url: string): Counts {
  const article = parse(readFileSync(`dist${url}index.html`, 'utf8')).querySelector(
    '.article-body',
  );
  if (!article) throw new Error(`no .article-body in ${url}`);
  return {
    images: article.querySelectorAll('img').length,
    math: article.querySelectorAll('.katex').length,
    citations: new Set(
      article.querySelectorAll('a[href^="#bib-"]').map((a) => a.getAttribute('href')),
    ).size,
    references: article.querySelectorAll('#refs .csl-entry').length,
  };
}

describe('post pages match the Jekyll site', () => {
  it.each(Object.keys(reference.posts))('%s', (url) => {
    expect(existsSync(`dist${url}index.html`)).toBe(true);
    const expected = { ...reference.posts[url], ...KNOWN_DIFFERENCES[url]?.counts };
    expect(builtCounts(url)).toEqual(expected);
    expect(readFileSync(`dist${url}index.html`, 'utf8')).not.toContain('katex-error');
  });

  it.each(Object.keys(reference.posts))('%s has no duplicate ids in the article', (url) => {
    const article = parse(readFileSync(`dist${url}index.html`, 'utf8')).querySelector('article')!;
    const ids = article.querySelectorAll('[id]').map((el) => el.id);
    expect(ids.filter((id, i) => ids.indexOf(id) !== i)).toEqual([]);
  });

  it('builds no post page that Jekyll did not publish', () => {
    const built = readdirSync('dist/posts', { recursive: true, encoding: 'utf8' })
      .filter((f) => f.endsWith('index.html'))
      .map((f) => `/posts/${f.replace(/index\.html$/, '')}`)
      .sort();
    expect(built).toEqual(Object.keys(reference.posts).sort());
  });
});
