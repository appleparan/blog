import { convertCite, removeBibliography } from './cite';
import { convertFrontmatter, usedComponents } from './frontmatter';
import { convertImg } from './img';
import { convertQuote } from './quote';
import { escapeMdx } from './escape';
import { convertYoutube } from './youtube';

export {
  convertCite,
  removeBibliography,
  convertImg,
  convertQuote,
  convertYoutube,
  escapeMdx,
  convertFrontmatter,
};

const FRONT_MATTER = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n?/;

/** Converts a Jekyll (kramdown + Liquid) post into MDX source. */
export function convertPost(source: string, fileName: string): string {
  const m = FRONT_MATTER.exec(source.replace(/\r\n/g, '\n'));
  if (!m) throw new Error(`${fileName}: missing front matter`);
  const raw = source.replace(/\r\n/g, '\n');

  // Escape first (Liquid tags are protected), then turn Liquid into components.
  let body = escapeMdx(removeBibliography(raw.slice(m[0].length)));
  const cite = convertCite(body);
  body = cite.body;
  const quote = convertQuote(body);
  body = convertYoutube(convertImg(quote.body));

  const files = [...new Set([...cite.files, ...quote.files])];
  if (files.length > 1) {
    throw new Error(`${fileName}: cites more than one bibliography file: ${files.join(', ')}`);
  }
  const header = convertFrontmatter(m[1], files[0] ?? null, usedComponents(body));
  return `${header}\n${body.replace(/^\n+/, '')}`;
}
