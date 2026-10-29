import { existsSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';
import { buildAtom } from '../src/lib/feed';
import type { Post } from '../src/lib/posts';

const post = (over: Partial<Post['data']> = {}): Post => ({
  id: '2024-03-05-Tom-and-Jerry',
  body: 'First paragraph of the body.',
  data: {
    title: 'Tom & <Jerry>',
    date: new Date('2024-03-05T00:00:00Z'),
    categories: [],
    tags: ['a b', 'c'],
    draft: false,
    ...over,
  },
});

describe('buildAtom', () => {
  const xml = buildAtom([post()], 'https://blog.liam.kim');

  it('escapes & and < in titles', () => {
    expect(xml).toContain('Tom &amp; &lt;Jerry&gt;');
    expect(xml).not.toContain('Tom & <Jerry>');
  });

  it('uses absolute URLs for id and link', () => {
    const url = 'https://blog.liam.kim/posts/2024/03/Tom-and-Jerry/';
    expect(xml).toContain(`<id>${url}</id>`);
    expect(xml).toContain(`<link href="${url}"`);
    expect(xml).toContain(
      '<link rel="self" type="application/atom+xml" href="https://blog.liam.kim/feed.xml"',
    );
  });

  it('has feed metadata, a summary, and a category per tag', () => {
    expect(xml).toContain('<title>Jongsu Liam Kim | Blog</title>');
    expect(xml).toContain('<updated>2024-03-05T00:00:00.000Z</updated>');
    expect(xml).toContain('<summary>First paragraph of the body.</summary>');
    expect(xml).toContain('<category term="a b"/>');
    expect(xml).toContain('<category term="c"/>');
  });

  it('uses the URL with its trailing slash as the id by default', () => {
    expect(xml).toContain('<id>https://blog.liam.kim/</id>');
  });

  it('writes Chirpy ids without a trailing slash and keeps the real URL in the link', () => {
    const chirpy = buildAtom([post()], 'https://blog.liam.kim', new Map(), {
      path: '/atom.xml',
      idStyle: 'chirpy',
    });
    expect(chirpy).toContain('<id>https://blog.liam.kim/posts/2024/03/Tom-and-Jerry</id>');
    expect(chirpy).toContain('<link href="https://blog.liam.kim/posts/2024/03/Tom-and-Jerry/"');
    expect(chirpy).toContain('<id>https://blog.liam.kim</id>');
    expect(chirpy).toContain('href="https://blog.liam.kim/atom.xml"');
  });
});

describe('built site', () => {
  it('serves the same entries at /feed.xml and /atom.xml, each with its old id style', () => {
    const feed = readFileSync('dist/feed.xml', 'utf8');
    const atom = readFileSync('dist/atom.xml', 'utf8');
    expect(feed.startsWith('<?xml')).toBe(true);
    expect(atom.startsWith('<?xml')).toBe(true);
    const links = (xml: string) => [...xml.matchAll(/<link href="([^"]+)"\/>/g)].map((m) => m[1]);
    const ids = (xml: string) => [...xml.matchAll(/<id>([^<]+)<\/id>/g)].map((m) => m[1]);
    expect(links(feed).length).toBeGreaterThan(0);
    expect(links(atom)).toEqual(links(feed));
    for (const url of links(feed)) expect(url).toMatch(/^https:\/\/blog\.liam\.kim\/posts\/.+\/$/);
    // jekyll-feed ids end with `/`; Chirpy ids (atom.xml) do not, and the feed id is the bare site.
    expect(ids(feed)).toEqual(['https://blog.liam.kim/', ...links(feed)]);
    expect(ids(atom)).toEqual([
      'https://blog.liam.kim',
      ...links(feed).map((url) => url.replace(/\/$/, '')),
    ]);
    expect(feed).toContain('href="https://blog.liam.kim/feed.xml"');
    expect(atom).toContain('href="https://blog.liam.kim/atom.xml"');
    // Apart from ids and the self link, the two feeds are the same.
    const strip = (xml: string) =>
      xml.replace(/<id>[^<]*<\/id>/g, '').replace(/\/(feed|atom)\.xml"/, '');
    expect(strip(atom)).toBe(strip(feed));
  });

  it('writes the sitemap index and a copy at /sitemap.xml', () => {
    expect(existsSync('dist/sitemap-index.xml')).toBe(true);
    expect(readFileSync('dist/sitemap.xml', 'utf8')).toBe(
      readFileSync('dist/sitemap-index.xml', 'utf8'),
    );
  });

  it('leaves the 404 page out of the sitemap', () => {
    const sitemap = readFileSync('dist/sitemap-0.xml', 'utf8');
    expect(sitemap).toContain('https://blog.liam.kim/about/');
    expect(sitemap).not.toMatch(/\/404(\.html|\/)/);
  });

  it('adds canonical, Open Graph, and the feed link to every page', () => {
    const root = parse(readFileSync('dist/about/index.html', 'utf8'));
    expect(root.querySelector('link[rel=canonical]')?.getAttribute('href')).toBe(
      'https://blog.liam.kim/about/',
    );
    expect(root.querySelector("meta[property='og:type']")?.getAttribute('content')).toBe('website');
    expect(root.querySelector("link[type='application/atom+xml']")?.getAttribute('href')).toBe(
      '/feed.xml',
    );
    expect(root.querySelector("script[type='application/ld+json']")).toBeNull();
  });

  it('adds BlogPosting JSON-LD and og:type article on posts', () => {
    const root = parse(readFileSync('dist/posts/2017/07/Tox-Workflow/index.html', 'utf8'));
    expect(root.querySelector("meta[property='og:type']")?.getAttribute('content')).toBe('article');
    const ld = JSON.parse(root.querySelector("script[type='application/ld+json']")!.text);
    expect(ld['@type']).toBe('BlogPosting');
    expect(ld.url).toBe('https://blog.liam.kim/posts/2017/07/Tox-Workflow/');
    expect(ld.datePublished).toMatch(/^2017-07/);
    expect(ld.dateModified).toBeTruthy();
    expect(ld.author.name).toBe('Jongsu Liam Kim');
  });

  it('leaves out the avatar as the link preview image on posts without an image', () => {
    const root = parse(readFileSync('dist/posts/2017/07/Tox-Workflow/index.html', 'utf8'));
    expect(root.querySelector("meta[property='og:image']")).toBeNull();
    expect(root.querySelector("meta[name='twitter:image']")).toBeNull();
    const ld = JSON.parse(root.querySelector("script[type='application/ld+json']")!.text);
    expect(ld.image).toBeUndefined();
  });

  it('keeps the avatar as the link preview image on non-post pages', () => {
    const root = parse(readFileSync('dist/about/index.html', 'utf8'));
    expect(root.querySelector("meta[property='og:image']")?.getAttribute('content')).toBe(
      'https://blog.liam.kim/assets/commons/avatar.jpg',
    );
  });
});
