import { existsSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';
import { siteConfig } from '../src/config';
import { draftPosts, homePagePath, pageCount, publishedPosts } from './helpers/source-posts';

// Reads the built site, so run `bun run build` before `bun run test`.
// Counts come from the post sources: `postsPerPage` per page, page 1 at `/`, then `/page/N/`.
const pages = Array.from({ length: pageCount }, (_, i) => homePagePath(i + 1));
const lastPage = pages[pages.length - 1];
const page = (path: string) => parse(readFileSync(`dist${path}index.html`, 'utf8'));
const cardUrls = (path: string) =>
  page(path)
    .querySelectorAll('.post-card .post-card-title a')
    .map((a) => a.getAttribute('href'));

describe('home page', () => {
  const allCards = () => pages.flatMap(cardUrls);

  it('fills every page but the last, and builds no page after the last', () => {
    expect(pageCount).toBeGreaterThan(0);
    for (const path of pages.slice(0, -1))
      expect(cardUrls(path), path).toHaveLength(siteConfig.postsPerPage);
    expect(cardUrls(lastPage).length).toBeGreaterThan(0);
    expect(cardUrls(lastPage).length).toBeLessThanOrEqual(siteConfig.postsPerPage);
    expect(existsSync(`dist/page/${pageCount + 1}/index.html`)).toBe(false);
  });

  it('lists every published post once, newest first', () => {
    expect(cardUrls('/')[0]).toBe(publishedPosts[0].url);
    expect(allCards()).toEqual(publishedPosts.map((p) => p.url));
  });

  it('shows no draft posts', () => {
    const cards = new Set(allCards());
    for (const draft of draftPosts) expect(cards.has(draft.url), draft.url).toBe(false);
  });

  it('shows meta and an excerpt on every card', () => {
    for (const card of page('/').querySelectorAll('.post-card')) {
      expect(card.querySelector('.post-card-meta')?.text).toMatch(/\d{4}-\d{2}-\d{2}.*\d+분/);
      expect(card.querySelector('.post-card-excerpt')?.text.length).toBeGreaterThan(0);
    }
  });

  it('lists 5 recently updated posts and 10 trending tags in the panel', () => {
    const panel = page('/').querySelector('.side-panel');
    expect(panel?.querySelectorAll('.side-recent a')).toHaveLength(5);
    const tags = panel?.querySelectorAll('.side-tags a') ?? [];
    expect(tags).toHaveLength(10);
    for (const a of tags) expect(a.getAttribute('href')).toMatch(/^\/tags\/[^/]+\/$/);
  });

  it('links pagination between the pages', () => {
    pages.forEach((path, i) => {
      const prev = page(path).querySelector('.pagination a[rel="prev"]')?.getAttribute('href');
      const next = page(path).querySelector('.pagination a[rel="next"]')?.getAttribute('href');
      expect(prev, path).toBe(pages[i - 1]);
      expect(next, path).toBe(pages[i + 1]);
    });
  });

  it('shows excerpts without holes left by removed citations or math', () => {
    const excerpts = pages.flatMap((path) =>
      page(path)
        .querySelectorAll('.post-card-excerpt')
        .map((p) => p.textContent),
    );
    expect(excerpts.length).toBeGreaterThan(0);
    expect(excerpts.filter((text) => text.includes('()') || text.includes('  '))).toEqual([]);
  });
});
