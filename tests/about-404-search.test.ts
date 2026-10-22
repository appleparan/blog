import { existsSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';
import { draftPosts, publishedPosts } from './helpers/source-posts';

// Reads the built site, so run `bun run build` before `bun run test`.
const html = (path: string) => readFileSync(`dist${path}`, 'utf8');
const searchData = (): { title: string; url: string }[] => {
  const root = parse(html('/index.html'));
  return JSON.parse(root.querySelector('#search-data')!.text);
};

describe('about page', () => {
  it('shows the avatar, the about text, and social links', () => {
    const root = parse(html('/about/index.html'));
    expect(root.querySelector('.about-avatar')?.getAttribute('src')).toBe(
      '/assets/commons/avatar.jpg',
    );
    expect(root.text).toContain('A Personal Blog on Machine Learning');
    const links = root.querySelectorAll('.about-links a').map((a) => a.getAttribute('href'));
    expect(links).toContain('https://github.com/appleparan');
    expect(links).toContain('mailto:jongsukim8@gmail.com');
  });
});

describe('404 page', () => {
  it('is built as /404.html and links home', () => {
    expect(existsSync('dist/404.html')).toBe(true);
    const hrefs = parse(html('/404.html'))
      .querySelectorAll('.not-found a')
      .map((a) => a.getAttribute('href'));
    expect(hrefs).toContain('/');
  });
});

describe('search data', () => {
  it('holds one entry per published post and no draft', () => {
    const entries = searchData();
    expect(entries).toHaveLength(publishedPosts.length);
    expect(entries.map((d) => d.url).sort()).toEqual(publishedPosts.map((p) => p.url).sort());
    const titles = entries.map((d) => d.title);
    for (const p of publishedPosts) expect(titles).toContain(p.data.title);
    for (const p of draftPosts) expect(titles).not.toContain(p.data.title);
  });

  it('links each entry to a post URL', () => {
    for (const d of searchData()) expect(d.url).toMatch(/^\/posts\/\d{4}\/\d{2}\/.+\/$/);
  });
});
