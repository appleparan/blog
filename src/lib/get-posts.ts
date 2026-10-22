import { getCollection } from 'astro:content';
import { compareByDate, type Post } from './posts';

/** Published posts, newest first (the reverse of `compareByDate`). Drafts are kept only in dev. */
export async function getPosts(): Promise<Post[]> {
  const entries = await getCollection('posts', ({ data }) => import.meta.env.DEV || !data.draft);
  return entries.sort((a, b) => compareByDate(b, a));
}
