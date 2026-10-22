import type { CollectionEntry } from 'astro:content';
import { siteConfig } from '../config';
import { buildAtom, type FeedOptions } from './feed';
import { getPosts } from './get-posts';
import { lastModified } from './lastmod';

/** The Atom feed response for `/feed.xml` and `/atom.xml`; they differ only in `options`. */
export async function atomResponse(options: FeedOptions): Promise<Response> {
  const posts = await getPosts();
  const lastmods = new Map(
    posts.map((p) => {
      const { filePath } = p as CollectionEntry<'posts'>;
      return [p.id, filePath ? lastModified(filePath) : null] as const;
    }),
  );
  return new Response(buildAtom(posts, siteConfig.url, lastmods, options), {
    headers: { 'Content-Type': 'application/atom+xml; charset=utf-8' },
  });
}
