import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';
import { publishedPosts } from './helpers/source-posts';

// Reads the built site, so run `bun run build` before `bun run test`.
const urls = publishedPosts.map((p) => p.url);

const article = (url: string) =>
  parse(readFileSync(`dist${url}index.html`, 'utf8')).querySelector('article');

describe('post pages', () => {
  it('builds a page for each published post source and no other post page', () => {
    const built = readdirSync('dist/posts', { recursive: true, encoding: 'utf8' })
      .filter((f) => f.endsWith('index.html'))
      .map((f) => `/posts/${f.replace(/index\.html$/, '')}`)
      .sort();
    expect(built).toEqual([...urls].sort());
  });

  it.each(urls)('%s renders without KaTeX errors', (url) => {
    expect(existsSync(`dist${url}index.html`)).toBe(true);
    expect(readFileSync(`dist${url}index.html`, 'utf8')).not.toContain('katex-error');
  });

  it.each(urls)('%s has no duplicate ids in the article', (url) => {
    const ids = article(url)!
      .querySelectorAll('[id]')
      .map((el) => el.id);
    expect(ids.filter((id, i) => ids.indexOf(id) !== i)).toEqual([]);
  });

  it.each(urls)('%s links every citation to a reference entry on the page', (url) => {
    const body = article(url)!;
    const entryIds = new Set(body.querySelectorAll('#refs .csl-entry').map((e) => e.id));
    const targets = body
      .querySelectorAll('a[href^="#bib-"]')
      .map((a) => decodeURIComponent(a.getAttribute('href')!.slice(1)));
    expect(targets.filter((id) => !entryIds.has(id))).toEqual([]);
  });
});

describe('site files', () => {
  it.each(['/404.html', '/robots.txt'])('%s exists', (file) => {
    expect(existsSync(`dist${file}`)).toBe(true);
  });
});
