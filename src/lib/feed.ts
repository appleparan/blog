import { siteConfig } from '../config';
import { excerpt } from './excerpt';
import { postUrl, type Post } from './posts';

const SUMMARY_LENGTH = 200;

/**
 * How entry and feed ids look. `url` uses the page URL as is (`/feed.xml`, as jekyll-feed did).
 * `chirpy` drops the trailing slash and uses the bare site as the feed id (`/atom.xml`, as the
 * Chirpy theme did), so existing subscribers do not see every entry as new.
 */
export type IdStyle = 'url' | 'chirpy';

export interface FeedOptions {
  /** Path of the feed itself, for the `self` link. */
  path: string;
  idStyle: IdStyle;
}

const DEFAULT_OPTIONS: FeedOptions = { path: '/feed.xml', idStyle: 'url' };

/** Escapes text for use in XML element content and attribute values. */
export function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Atom 1.0 feed for `posts` (expected newest first). `lastmods` maps a post id to its last
 * edit date; a post without one is `updated` at its `date`.
 */
export function buildAtom(
  posts: Post[],
  site: string,
  lastmods: ReadonlyMap<string, Date | null> = new Map(),
  { path, idStyle }: FeedOptions = DEFAULT_OPTIONS,
): string {
  const abs = (p: string) => new URL(p, site).href;
  const idOf = (url: string) => (idStyle === 'chirpy' ? url.replace(/\/+$/, '') : url);
  const updatedOf = (post: Post) => lastmods.get(post.id) ?? post.data.date;
  const feedUpdated = posts.reduce<Date | null>((latest, p) => {
    const d = updatedOf(p);
    return !latest || d > latest ? d : latest;
  }, null);

  const entries = posts.map((post) => {
    const url = abs(postUrl(post));
    const summary = post.data.description ?? excerpt(post.body ?? '', SUMMARY_LENGTH);
    return [
      '  <entry>',
      `    <title>${escapeXml(post.data.title)}</title>`,
      `    <link href="${escapeXml(url)}"/>`,
      `    <id>${escapeXml(idOf(url))}</id>`,
      `    <published>${post.data.date.toISOString()}</published>`,
      `    <updated>${updatedOf(post).toISOString()}</updated>`,
      summary ? `    <summary>${escapeXml(summary)}</summary>` : '',
      ...post.data.tags.map((t) => `    <category term="${escapeXml(t)}"/>`),
      '  </entry>',
    ]
      .filter(Boolean)
      .join('\n');
  });

  return [
    '<?xml version="1.0" encoding="utf-8"?>',
    `<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="${siteConfig.lang}">`,
    `  <title>${escapeXml(siteConfig.title)}</title>`,
    `  <subtitle>${escapeXml(siteConfig.description)}</subtitle>`,
    `  <link rel="self" type="application/atom+xml" href="${abs(path)}"/>`,
    `  <link rel="alternate" type="text/html" href="${abs('/')}"/>`,
    `  <id>${idOf(abs('/'))}</id>`,
    `  <updated>${(feedUpdated ?? new Date(0)).toISOString()}</updated>`,
    `  <author><name>${escapeXml(siteConfig.author)}</name></author>`,
    ...entries,
    '</feed>',
    '',
  ].join('\n');
}
