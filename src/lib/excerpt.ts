const ELLIPSIS = '…';

/** Block-level syntax that never counts as prose. */
const NON_PROSE_START = /^(#|import\s|export\s|<|\$\$|\||>|[-*+]\s|\d+[.)]\s|!\[|---|\{%|\{\/?\*)/;

function stripBlocks(body: string): string {
  return (
    body
      .replace(/^(```|~~~)[\s\S]*?^\1[^\n]*$/gm, '')
      .replace(/\$\$[\s\S]*?\$\$/g, '')
      .replace(/<!--[\s\S]*?-->/g, '')
      // A heading line may sit directly on top of its paragraph with no blank line between.
      .replace(/^#{1,6}\s.*$/gm, '')
      .replace(/<([A-Z][\w.]*)\b[^>]*?\/>/g, '')
      .replace(/<([A-Z][\w.]*)\b[\s\S]*?<\/\1>/g, '')
  );
}

function stripInline(text: string): string {
  // Set inline math aside so the Markdown rules below cannot touch its TeX (`x_i`, `a<b`).
  const math: string[] = [];
  const hidden = text.replace(
    /\$([^$\n]+)\$/g,
    (_, tex: string) => `\uE000${math.push(tex.replace(/\s+/g, ' ').trim()) - 1}\uE000`,
  );
  return hidden
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[@[^\]]*\]/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\{%[\s\S]*?%\}/g, '')
    .replace(/\{\{[\s\S]*?\}\}/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/\(\s*\)/g, '')
    .replace(/\s+/g, ' ')
    .replace(/ ([,.])/g, '$1')
    .replace(/\uE000(\d+)\uE000/g, (_, i: string) => math[Number(i)])
    .trim();
}

const CITATION = /\[@[^\]]*\]/;

/**
 * The block without the sentences that cite a source. A removed citation is often the
 * subject of its sentence ("[@key]에서도 ..."), so dropping only the citation leaves a hole.
 */
function withoutCitedSentences(block: string): string {
  return block
    .split(/(?<=[.!?])\s+/)
    .filter((sentence) => !CITATION.test(sentence))
    .join(' ');
}

function cut(text: string, max: number): string {
  return text.length > max ? text.slice(0, max - 1).trimEnd() + ELLIPSIS : text;
}

/**
 * Plain-text summary of an MDX post body: its first prose paragraph with Markdown, MDX,
 * and Liquid syntax removed, cut to `max` characters (the last one is `…` when cut).
 * Inline math keeps its TeX source without the `$` delimiters. Sentences with a citation
 * are left out; when every prose sentence cites, the first paragraph is used with its
 * citations removed. Returns an empty string when the body has no prose paragraph.
 */
export function excerpt(body: string, max = 150): string {
  const blocks = stripBlocks(body)
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter((block) => block && !NON_PROSE_START.test(block));
  for (const block of blocks) {
    const text = stripInline(withoutCitedSentences(block));
    if (text) return cut(text, max);
  }
  for (const block of blocks) {
    const text = stripInline(block);
    if (text) return cut(text, max);
  }
  return '';
}
