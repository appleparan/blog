import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
const html = (path: string) => readFileSync(`dist${path}`, 'utf8');
const frontMatter = (file: string) =>
  readFileSync(`src/content/posts/${file}`, 'utf8').split(/^---$/m)[1] ?? '';
const titleOf = (file: string) =>
  frontMatter(file)
    .match(/^title:\s*(.+)$/m)?.[1]
    ?.trim()
    .replace(/^(["'])(.*)\1$/, '$2') ?? '';
const isDraft = (file: string) => /^draft:\s*true\s*$/m.test(frontMatter(file));

const postFiles = readdirSync('src/content/posts').filter((f) => /\.mdx?$/.test(f));
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
  it('holds every published post title and no draft title', () => {
    const titles = searchData().map((d) => d.title);
    const published = postFiles.filter((f) => !isDraft(f)).map(titleOf);
    const drafts = postFiles.filter(isDraft).map(titleOf);
    expect(published).toHaveLength(20);
    for (const t of published) expect(titles).toContain(t);
    for (const t of drafts) expect(titles).not.toContain(t);
  });

  it('links each entry to a post URL', () => {
    for (const d of searchData()) expect(d.url).toMatch(/^\/posts\/\d{4}\/\d{2}\/.+\/$/);
  });
});
