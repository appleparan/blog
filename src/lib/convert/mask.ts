// Placeholder masking: hide spans from regex passes, then put them back.
// Private-use characters (U+E000, U+E001) never occur in posts, so they are safe delimiters.
const OPEN = '\uE000';
const CLOSE = '\uE001';
const PLACEHOLDER = new RegExp(`${OPEN}(\\d+)${CLOSE}`, 'g');

export interface Masked {
  text: string;
  restore: (s: string) => string;
}

/** A pattern, optionally with a function that rewrites each match before it is stored. */
export type MaskPattern =
  | RegExp
  | [RegExp, (match: string, whole: string, offset: number) => string];

/** Replaces every match of each pattern (in order) with a placeholder. */
export function maskPatterns(text: string, patterns: MaskPattern[]): Masked {
  const store: string[] = [];
  let out = text;
  for (const pattern of patterns) {
    const [re, rewrite] = Array.isArray(pattern) ? pattern : [pattern, undefined];
    out = out.replace(re, (m, ...rest) => {
      const whole = rest[rest.length - 1] as string;
      const offset = rest[rest.length - 2] as number;
      store.push(rewrite ? rewrite(m, whole, offset) : m);
      return `${OPEN}${store.length - 1}${CLOSE}`;
    });
  }
  const restore = (s: string): string => {
    let cur = s;
    // Masked spans can contain earlier placeholders, so unwrap until none are left.
    while (PLACEHOLDER.test(cur)) {
      PLACEHOLDER.lastIndex = 0;
      cur = cur.replace(PLACEHOLDER, (_, i: string) => store[Number(i)]);
    }
    PLACEHOLDER.lastIndex = 0;
    return cur;
  };
  return { text: out, restore };
}

/** Fenced code blocks (indented fences allowed, e.g. inside list items). */
export const FENCE = /^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^[ \t]*\1[`~]*[ \t]*$|(?![\s\S]))/gm;
/** Inline code spans. */
export const INLINE_CODE = /(`+)(?!`)[^\n]*?(?<!`)\1(?!`)/g;

/** Runs `fn` on text with fenced and inline code hidden. */
export function outsideCode(body: string, fn: (text: string) => string): string {
  const { text, restore } = maskPatterns(body, [FENCE, INLINE_CODE]);
  return restore(fn(text));
}
