import type { Converted } from './cite';
import { outsideCode } from './mask';

const QUOTE =
  /\{%\s*quote\s+(\S+)\s+--file\s+(\S+?)\s*%\}\n?([\s\S]*?)\n?[ \t]*\{%\s*endquote\s*%\}[ \t]*/g;

/** `{% quote key --file F %}…{% endquote %}` becomes a `<Quote>` block. */
export function convertQuote(body: string): Converted {
  const files: string[] = [];
  const out = outsideCode(body, (text) =>
    text.replace(QUOTE, (_, key: string, file: string, inner: string) => {
      files.push(file);
      // Blank lines let MDX parse the inside as markdown.
      return `<Quote cite="${key}">\n\n${inner.trim()}\n\n</Quote>`;
    }),
  );
  return { body: out, files: [...new Set(files)] };
}
