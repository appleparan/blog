import { siteConfig } from "../config";

/** Minimal structural post type, compatible with CollectionEntry<'posts'>. */
export interface Post {
  id: string;
  body?: string;
  data: {
    title: string;
    date: Date;
    categories: string[];
    tags: string[];
    draft: boolean;
    author?: string;
    description?: string;
    image?: string;
    math?: boolean;
    mermaid?: boolean;
    bibliography?: string;
  };
}

export interface NamedPosts {
  name: string;
  posts: Post[];
}

export interface CategoryNode {
  name: string;
  slug: string;
  count: number;
}

/** Jekyll-compatible slug: drop punctuation, spaces to hyphens, lowercase. */
export function slugify(name: string): string {
  return name
    .replace(/[^\p{L}\p{M}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

/** Strips the `YYYY-MM-DD-` prefix from a post id. */
export function postSlug(id: string): string {
  return id.replace(/^\d{4}-\d{2}-\d{2}-/, "");
}

const seoulFormat = new Intl.DateTimeFormat("en-CA", {
  timeZone: siteConfig.timezone,
  year: "numeric",
  month: "2-digit",
});

export function seoulYearMonth(date: Date): { year: string; month: string } {
  const parts = seoulFormat.formatToParts(date);
  const pick = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return { year: pick("year"), month: pick("month") };
}

export function postUrl(post: { id: string; data: { date: Date } }): string {
  const { year, month } = seoulYearMonth(post.data.date);
  return `/posts/${year}/${month}/${postSlug(post.id)}/`;
}

function groupBy(posts: Post[], names: (p: Post) => string[]): Map<string, NamedPosts> {
  const map = new Map<string, NamedPosts>();
  for (const post of posts) {
    for (const name of names(post)) {
      const slug = slugify(name);
      const entry = map.get(slug);
      if (!entry) map.set(slug, { name, posts: [post] });
      else if (!entry.posts.includes(post)) entry.posts.push(post);
    }
  }
  return map;
}

/** One entry per category at any level, keyed by slug. */
export function categoryIndex(posts: Post[]): Map<string, NamedPosts> {
  return groupBy(posts, (p) => p.data.categories);
}

export function tagIndex(posts: Post[]): Map<string, NamedPosts> {
  return groupBy(posts, (p) => p.data.tags);
}

/** Parent is `categories[0]`, children are `categories[1]`. */
export function categoryTree(posts: Post[]): (CategoryNode & { children: CategoryNode[] })[] {
  const parents = new Map<
    string,
    { name: string; posts: Set<Post>; kids: Map<string, NamedPosts> }
  >();
  for (const post of posts) {
    const [top, sub] = post.data.categories;
    if (!top) continue;
    const key = slugify(top);
    const parent = parents.get(key) ?? { name: top, posts: new Set(), kids: new Map() };
    parents.set(key, parent);
    parent.posts.add(post);
    if (sub) {
      const kid = parent.kids.get(slugify(sub)) ?? { name: sub, posts: [] };
      kid.posts.push(post);
      parent.kids.set(slugify(sub), kid);
    }
  }
  return [...parents.entries()]
    .map(([slug, p]) => ({
      name: p.name,
      slug,
      count: p.posts.size,
      children: [...p.kids.entries()]
        .map(([s, k]) => ({ name: k.name, slug: s, count: k.posts.length }))
        .sort((a, b) => a.name.localeCompare(b.name)),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function trendingTags(
  posts: Post[],
  limit = 10,
): { name: string; slug: string; count: number }[] {
  return [...tagIndex(posts).entries()]
    .map(([slug, t]) => ({ name: t.name, slug, count: t.posts.length }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, limit);
}

export function relatedPosts(post: Post, posts: Post[], limit = 3): Post[] {
  const tags = new Set(post.data.tags);
  const cats = new Set(post.data.categories);
  return posts
    .filter((p) => p.id !== post.id)
    .map((p) => ({
      p,
      score:
        new Set(p.data.tags.filter((t) => tags.has(t))).size +
        new Set(p.data.categories.filter((c) => cats.has(c))).size,
    }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || b.p.data.date.getTime() - a.p.data.date.getTime())
    .slice(0, limit)
    .map((x) => x.p);
}

/** `prev` is the next older post, `next` the next newer one. */
export function adjacentPosts(post: Post, posts: Post[]): { prev: Post | null; next: Post | null } {
  const sorted = [...posts].sort((a, b) => a.data.date.getTime() - b.data.date.getTime());
  const i = sorted.findIndex((p) => p.id === post.id);
  if (i < 0) return { prev: null, next: null };
  return { prev: sorted[i - 1] ?? null, next: sorted[i + 1] ?? null };
}
