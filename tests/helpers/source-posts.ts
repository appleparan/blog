import { readdirSync, readFileSync } from 'node:fs';
import { parse } from 'yaml';
import { siteConfig } from '../../src/config';
import { compareByDate, postUrl, slugify, type Post } from '../../src/lib/posts';

// Reads the post sources, so tests derive counts from the posts instead of pinning them.
const POSTS_DIR = 'src/content/posts';

export interface SourcePost extends Post {
  url: string;
}

function readPost(file: string): SourcePost {
  const yaml = readFileSync(`${POSTS_DIR}/${file}`, 'utf8').split(/^---$/m)[1] ?? '';
  const fm = parse(yaml) as Record<string, unknown>;
  const date = new Date(String(fm.date));
  if (Number.isNaN(date.getTime())) throw new Error(`${file}: bad date ${String(fm.date)}`);
  const data: Post['data'] = {
    title: String(fm.title),
    date,
    categories: ((fm.categories as unknown[] | undefined) ?? []).map(String),
    tags: ((fm.tags as unknown[] | undefined) ?? []).map(String),
    draft: fm.draft === true,
  };
  const post = { id: file.replace(/\.mdx$/, ''), data };
  return { ...post, url: postUrl(post) };
}

const all = readdirSync(POSTS_DIR)
  .filter((f) => f.endsWith('.mdx'))
  .map(readPost)
  .sort((a, b) => compareByDate(b, a));

/** Posts the build publishes (no `draft: true`), newest first, in the site's order. */
export const publishedPosts: readonly SourcePost[] = all.filter((p) => !p.data.draft);

/** `draft: true` posts, which only `bun run dev` shows. */
export const draftPosts: readonly SourcePost[] = all.filter((p) => p.data.draft);

/** Number of home pages: `/` plus `/page/N/` for N = 2..pageCount. */
export const pageCount = Math.ceil(publishedPosts.length / siteConfig.postsPerPage);

/** Home page path for page `n` (1-based). */
export const homePagePath = (n: number) => (n === 1 ? '/' : `/page/${n}/`);

/** Category and tag page URLs the published posts need. */
export function taxonomyUrls(kind: 'categories' | 'tags'): string[] {
  const names = publishedPosts.flatMap((p) => p.data[kind]);
  return [...new Set(names.map((n) => `/${kind}/${slugify(n)}/`))].sort();
}
