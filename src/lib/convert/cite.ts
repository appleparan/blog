import { outsideCode } from './mask';

export interface Converted {
  body: string;
  /** Bibliography files named by the converted tags. */
  files: string[];
}

const CITE = /\{%\s*cite\s+((?:(?!--file)[^%])*?)\s*--file\s+(\S+?)\s*%\}/g;
const BIBLIOGRAPHY_TAG = /\{%\s*bibliography\b[^%]*%\}/;

/** `{% cite a b --file F %}` becomes `[@a; @b]`. */
export function convertCite(body: string): Converted {
  const files: string[] = [];
  const out = outsideCode(body, (text) =>
    text.replace(CITE, (_, keys: string, file: string) => {
      files.push(file);
      return `[${keys
        .trim()
        .split(/\s+/)
        .map((k) => `@${k}`)
        .join('; ')}]`;
    }),
  );
  return { body: out, files: [...new Set(files)] };
}

const HEADING_ONLY = String.raw`(?:^#{1,6}[ \t]*(?:References?|참고문헌)[ \t]*\n(?:[ \t]*\n)*)?`;
const BIBLIOGRAPHY_LINE = new RegExp(
  `${HEADING_ONLY}^[ \\t]*${BIBLIOGRAPHY_TAG.source}[ \\t]*(?:\\n|$)`,
  'gm',
);

/** Drops `{% bibliography %}` lines and a bare References heading right before one. */
export function removeBibliography(body: string): string {
  return outsideCode(body, (text) => text.replace(BIBLIOGRAPHY_LINE, ''));
}
