import { getCollection } from 'astro:content';
import type { Post } from './posts';

/** Published posts, newest first. Drafts are kept only in dev. */
export async function getPosts(): Promise<Post[]> {
  const entries = await getCollection('posts', ({ data }) => import.meta.env.DEV || !data.draft);
  return entries.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}
