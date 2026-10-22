const ELLIPSIS = '…';

/** Block-level syntax that never counts as prose. */
const NON_PROSE_START = /^(#|import\s|export\s|<|\$\$|\||>|[-*+]\s|\d+[.)]\s|!\[|---|\{%|\{\/?\*)/;

function stripBlocks(body: string): string {
  return body
    .replace(/^(```|~~~)[\s\S]*?^\1[^\n]*$/gm, '')
    .replace(/\$\$[\s\S]*?\$\$/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<([A-Z][\w.]*)\b[^>]*?\/>/g, '')
    .replace(/<([A-Z][\w.]*)\b[\s\S]*?<\/\1>/g, '');
}

function stripInline(text: string): string {
  return text
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[@[^\]]*\]/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\{%[\s\S]*?%\}/g, '')
    .replace(/\{\{[\s\S]*?\}\}/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\$[^$\n]+\$/g, '')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Plain-text summary of an MDX post body: its first prose paragraph with Markdown, MDX,
 * math, and Liquid syntax removed, cut to `max` characters (the last one is `…` when cut).
 * Returns an empty string when the body has no prose paragraph.
 */
export function excerpt(body: string, max = 150): string {
  for (const paragraph of stripBlocks(body).split(/\n\s*\n/)) {
    const block = paragraph.trim();
    if (!block || NON_PROSE_START.test(block)) continue;
    const text = stripInline(block);
    if (!text) continue;
    return text.length > max ? text.slice(0, max - 1).trimEnd() + ELLIPSIS : text;
  }
  return '';
}
