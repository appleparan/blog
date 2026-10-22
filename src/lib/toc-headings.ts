import katex from 'katex';

export interface TocHeading {
  depth: number;
  slug: string;
  text: string;
  /** Heading content with rendered math, when the heading has math. */
  html?: string;
}

const FENCE = /^\s*(```|~~~)/;
const ATX = /^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$/;
const MATH = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g;
const hasMath = (s: string): boolean => /\$[^$]+\$/.test(s);

/** ATX headings of a Markdown body, skipping fenced code. */
function sourceHeadings(body: string): { depth: number; text: string }[] {
  const out: { depth: number; text: string }[] = [];
  let fence: string | null = null;
  for (const line of body.split('\n')) {
    const f = FENCE.exec(line)?.[1];
    if (f && (fence === null || fence === f)) {
      fence = fence === null ? f : null;
      continue;
    }
    if (fence !== null) continue;
    const m = ATX.exec(line);
    if (m) out.push({ depth: m[1].length, text: m[2] });
  }
  return out;
}

const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function renderMath(source: string): { text: string; html: string } {
  let text = '';
  let html = '';
  let last = 0;
  for (const m of source.matchAll(MATH)) {
    const tex = (m[1] ?? m[2]).trim();
    const plain = source.slice(last, m.index);
    text += plain + tex;
    html += escapeHtml(plain) + katex.renderToString(tex, { throwOnError: false, output: 'html' });
    last = m.index + m[0].length;
  }
  const rest = source.slice(last);
  return { text: text + rest, html: html + escapeHtml(rest) };
}

/**
 * Astro collects a heading's text from every text node, so a heading with KaTeX math reads
 * as MathML text, TeX, and glyphs run together. This takes the TeX from the post source
 * instead and renders it. Headings are matched by order and depth; the only extra heading
 * allowed is the References heading that follows the body. If they do not line up, the
 * headings are returned unchanged.
 */
export function tocHeadings(headings: TocHeading[], body: string): TocHeading[] {
  const source = sourceHeadings(body);
  if (!source.some((h) => hasMath(h.text))) return headings;
  const extra = headings.slice(source.length);
  const lineUp =
    extra.every((h) => h.slug === 'references') &&
    source.every((s, i) => headings[i]?.depth === s.depth);
  if (!lineUp) return headings;
  return headings.map((h, i) => {
    const s = source[i];
    if (!s || !hasMath(s.text)) return h;
    return { ...h, ...renderMath(s.text) };
  });
}
