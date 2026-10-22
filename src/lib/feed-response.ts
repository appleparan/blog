import type { CollectionEntry } from 'astro:content';
import { siteConfig } from '../config';
import { buildAtom } from './feed';
import { getPosts } from './get-posts';
import { lastModified } from './lastmod';

/** The Atom feed response shared by `/feed.xml` and `/atom.xml`. */
export async function atomResponse(): Promise<Response> {
  const posts = await getPosts();
  const lastmods = new Map(
    posts.map((p) => {
      const { filePath } = p as CollectionEntry<'posts'>;
      return [p.id, filePath ? lastModified(filePath) : null] as const;
    }),
  );
  return new Response(buildAtom(posts, siteConfig.url, lastmods), {
    headers: { 'Content-Type': 'application/atom+xml; charset=utf-8' },
  });
}
