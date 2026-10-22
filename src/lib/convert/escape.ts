import { FENCE, INLINE_CODE, maskPatterns } from './mask';

const LIQUID = /\{%[\s\S]*?%\}/g;
const DISPLAY_MATH = /\$\$[\s\S]*?\$\$/g;

// Kramdown's `\\( … \\)` inline math. remark-math only knows `$`, so it becomes `$…$`.
const PAREN_MATH = /\\\\\(.+?\\\\\)/g;
const parenToDollar = (m: string): string => `$${m.slice(3, -3)}$`;

// A `$$` pair is a display block only when both `$$` sit on their own lines. Otherwise kramdown
// reads it as inline math, but remark-math would read a line-start `$$x` as a block fence.
// Such a pair becomes `$x$`, which renders the same inline math.
function fixInlineDisplayMath(match: string, whole: string, offset: number): string {
  const end = offset + match.length;
  const isBlock =
    /^\$\$[ \t]*\n/.test(match) &&
    /\n[ \t]*\$\$$/.test(match) &&
    /^[ \t]*(\n|$)/.test(whole.slice(end));
  const atLineStart = offset === 0 || whole[offset - 1] === '\n';
  if (isBlock || !atLineStart) return match;
  return `$${match.slice(2, -2)}$`;
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
