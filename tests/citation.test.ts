import { readFileSync, readdirSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
const URL = '/posts/2024/05/Tensor-Program-2/';
const page = `dist${URL}index.html`;

describe(`citations on ${URL}`, () => {
  const html = readFileSync(page, 'utf8');
  const article = parse(html).querySelector('.article-body');

  it('renders the article body', () => {
    expect(article).not.toBeNull();
  });

  it('lists exactly the cited keys as references', () => {
    const cited = new Set(
      article!
        .querySelectorAll('a[href^="#bib-"]')
        .filter((a) => !a.closest('#refs'))
        .map((a) => a.getAttribute('href')!.slice(1)),
    );
    const listed = article!.querySelectorAll('#refs .csl-entry').map((e) => e.id);
    expect(cited.size).toBeGreaterThan(0);
    expect([...cited].sort()).toEqual([...listed].sort());
  });

  it('renders math with KaTeX', () => {
    expect(article!.querySelectorAll('.katex').length).toBeGreaterThan(0);
  });
});

describe(`citation back-links and popover on ${URL}`, () => {
  const html = readFileSync(page, 'utf8');
  const root = parse(html);
  const ids = new Set(root.querySelectorAll('[id]').map((n) => n.getAttribute('id')));
  const entries = root.querySelectorAll('#refs .csl-entry');
  const cited = new Set(
    root
      .querySelectorAll('.article-body a[href^="#bib-"]')
      .map((a) => a.getAttribute('href')!.slice(1)),
  );

  it('gives every cited entry back-links to ids on the same page', () => {
    for (const entry of entries.filter((e) => cited.has(e.getAttribute('id')!))) {
      const links = entry.querySelectorAll('a.backlink');
      expect(links.length, entry.getAttribute('id')!).toBeGreaterThan(0);
      for (const link of links) {
        expect(link.getAttribute('aria-label')).toBe('본문으로 돌아가기');
        expect(ids.has(link.getAttribute('href')!.slice(1))).toBe(true);
      }
    }
  });

  it('gives back-links only to entries that the text cites (noCite entries get none)', () => {
    const noCite = parse(
      readFileSync('dist/posts/2024/03/Logit-Sigmoid-Softmax/index.html', 'utf8'),
    );
    const inText = new Set(
      noCite
        .querySelectorAll('.article-body a[href^="#bib-"]')
        .filter((a) => !a.closest('#refs'))
        .map((a) => a.getAttribute('href')!.slice(1)),
    );
    const list = noCite.querySelectorAll('#refs .csl-entry');
    const uncited = list.filter((e) => !inText.has(e.getAttribute('id')!));
    expect(uncited.length).toBeGreaterThan(0);
    for (const entry of list) {
      const count = entry.querySelectorAll('a.backlink').length;
      expect(count > 0).toBe(inText.has(entry.getAttribute('id')!));
    }
  });

  it('ships the popover script with its guard and selector', () => {
    const scripts = root.querySelectorAll('script').map((s) => {
      const src = s.getAttribute('src');
      return src ? readFileSync(`dist${src}`, 'utf8') : s.text;
    });
    const code = scripts.find((c) => c.includes('cite-popover'));
    expect(code).toBeDefined();
    expect(code).toMatch(/getElementById\(["'`]refs["'`]\)/);
    expect(code).toMatch(/a\[href\^=["']#bib-/);
  });

  it('ships no unprocessed :global() in built CSS', () => {
    const dir = 'dist/_astro';
    const css = readdirSync(dir).filter((f) => f.endsWith('.css'));
    expect(css.length).toBeGreaterThan(0);
    for (const f of css) expect(readFileSync(`${dir}/${f}`, 'utf8'), f).not.toContain(':global(');
  });
});
