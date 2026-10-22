// Usage: bun scripts/capture-reference.ts <siteDir> <outFile> <postsDir>
// Records every URL and per-post media counts of a built Jekyll site, so the Astro build can be compared to it.
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { parse, type HTMLElement } from 'node-html-parser';

const [siteDir, outFile, postsDir] = process.argv.slice(2);
if (!siteDir || !outFile || !postsDir) {
  throw new Error('usage: capture-reference.ts <siteDir> <outFile> <postsDir>');
}

const POSTS_PER_PAGE = 10;

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

const files = walk(siteDir).map((f) => relative(siteDir, f).split(sep).join('/'));
const siteUrls = files
  .filter((f) => f.endsWith('index.html') || f.endsWith('.xml'))
  .map((f) => (f.endsWith('index.html') ? '/' + f.slice(0, -'index.html'.length) : '/' + f))
  .sort();

// _site can be stale, so keep only generated pages that the current posts produce.
const slugify = (name: string): string => name.trim().toLowerCase().replace(/[^\p{L}\p{N}_-]+/gu, '-').replace(/^-+|-+$/g, '');
const expected = new Set<string>();
let postCount = 0;
for (const file of readdirSync(postsDir).filter((f) => /^\d{4}-\d{2}-\d{2}-.+\.md$/.test(f))) {
  const slug = file.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
  const front = Bun.YAML.parse(readFileSync(join(postsDir, file), 'utf8').split(/^---\s*$/m)[1]) as {
    date: string | Date;
    draft?: boolean;
    categories?: string[];
    tags?: string[];
  };
  if (front.draft === true) continue; // drafts are not published
  const date = new Date(front.date);
  // Jekyll builds the URL from the front matter date in the site timezone (+0900 here).
  const local = new Date(date.getTime() + 9 * 3600 * 1000).toISOString();
  expected.add(`/posts/${local.slice(0, 4)}/${local.slice(5, 7)}/${slug}/`);
  for (const c of front.categories ?? []) expected.add(`/categories/${slugify(c)}/`);
  for (const t of front.tags ?? []) expected.add(`/tags/${slugify(String(t))}/`);
  postCount += 1;
}
for (let n = 2; n <= Math.ceil(postCount / POSTS_PER_PAGE); n++) expected.add(`/page/${n}/`);

const isGenerated = (u: string): boolean => /^\/(posts\/\d{4}\/\d{2}|categories|tags|page)\/[^/]+\/$/.test(u);
const urls = siteUrls.filter((u) => !isGenerated(u) || expected.has(u));
const dropped = siteUrls.filter((u) => !urls.includes(u));
const missing = [...expected].filter((u) => !siteUrls.includes(u)).sort();
console.log('dropped (in _site, not produced by current posts):', dropped);
console.log('missing (produced by current posts, not in _site):', missing);

function count(html: string, pattern: RegExp): number {
  return (html.match(pattern) ?? []).length;
}

// MathJax delimiters left literal by kramdown: `$$…$$` (display) and `$…$` (inline).
const DOLLAR_MATH = /\$\$[\s\S]+?\$\$|\$[^$]+?\$/g;

/** Text MathJax scans for `$` math: the article without `code` and `pre`. */
function mathText(article: HTMLElement): string {
  const copy = parse(article.outerHTML);
  for (const el of copy.querySelectorAll('pre, code')) el.remove();
  return copy.textContent;
}

const posts: Record<string, { images: number; math: number; citations: number; references: number }> = {};
for (const url of urls.filter((u) => /^\/posts\/\d{4}\/\d{2}\/[^/]+\/$/.test(u))) {
  const root = parse(readFileSync(join(siteDir, url, 'index.html'), 'utf8'));
  const article = root.querySelector('article .content');
  if (!article) throw new Error(`no article content in ${url}`);
  const html = article.innerHTML;
  const targets = new Set(article.querySelectorAll('a.citation').map((a) => a.getAttribute('href')));
  posts[url] = {
    images: article.querySelectorAll('img').length,
    // Chirpy renders math client-side with MathJax: kramdown's \( and \[ openers plus the
    // `$…$` spans kramdown left literal, which MathJax also renders.
    math: count(html, /\\\(|\\\[/g) + count(mathText(article), DOLLAR_MATH),
    citations: targets.size,
    references: article.querySelectorAll('ol.bibliography li').length,
  };
}

writeFileSync(outFile, JSON.stringify({ urls, posts }, null, 2) + '\n');
console.log(`${urls.length} urls, ${Object.keys(posts).length} posts`);
