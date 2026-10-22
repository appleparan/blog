import { FENCE, INLINE_CODE, maskPatterns } from './mask';

const LIQUID = /\{%[\s\S]*?%\}/g;
const DISPLAY_MATH = /\$\$[\s\S]*?\$\$/g;

// Kramdown's `\\( … \\)` inline math. remark-math only knows `$`, so it becomes `$…$`.
const PAREN_MATH = /\\\\\(.+?\\\\\)/g;
const parenToDollar = (m: string): string => `$${m.slice(3, -3)}$`;

// How a `$$…$$` span is read:
// - own lines (`$$` line, body, `$$` line): display block, kept as is.
// - one paragraph on its own (blank line or file edge on both sides): display math, so a
//   single-line span becomes block form (the leading indent is kept).
// - anything else is inline math. remark-math would read a line-start `$$x` as a block fence,
//   so such a span becomes `$x$`, which renders the same inline math.
function fixInlineDisplayMath(match: string, whole: string, offset: number): string {
  const lineStart = whole.lastIndexOf('\n', offset - 1) + 1;
  const indent = whole.slice(lineStart, offset);
  if (/\S/.test(indent)) return match; // mid-sentence inline math
  const rest = whole.slice(offset + match.length);
  const endsLine = /^[ \t]*(\n|$)/.test(rest);
  const isBlock = /^\$\$[ \t]*\n/.test(match) && /\n[ \t]*\$\$$/.test(match) && endsLine;
  if (isBlock) return match;
  const blankBefore = lineStart === 0 || /(^|\n)[ \t]*\n$/.test(whole.slice(0, lineStart));
  const blankAfter = endsLine && /^[ \t]*(\n[ \t]*(\n|$)|$)/.test(rest);
  const inner = match.slice(2, -2);
  if (blankBefore && blankAfter) {
    return `$$\n${indent}${inner.trim()}\n${indent}$$`;
  }
  return `$${inner}$`;
}

// Single-dollar math on one line. The closing `$` is not followed by a digit, so "$5 and $10" stays text.
const INLINE_MATH = /\$(?!\$)(?:\\.|[^$\\\n])+?\$(?![\d$])/g;

// Tags that may stay as JSX. Every other `<` becomes text.
const NOT_KNOWN_TAG = /<(?!\/?(?:a|br|Figure|Quote|YouTube)(?=[\s/>]))/g;

/** Escapes MDX-special characters in prose; code, math and Liquid tags stay as they are. */
export function escapeMdx(body: string): string {
  const { text, restore } = maskPatterns(body, [
    FENCE,
    INLINE_CODE,
    LIQUID,
    [PAREN_MATH, parenToDollar],
    [DISPLAY_MATH, fixInlineDisplayMath],
    INLINE_MATH,
  ]);
  const escaped = text
    .replace(/<br\s*>/g, '<br />')
    .replace(/[{}]/g, (c) => `\\${c}`)
    .replace(NOT_KNOWN_TAG, '&lt;');
  return restore(escaped);
}
