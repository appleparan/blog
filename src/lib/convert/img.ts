import { outsideCode } from './mask';

const IMG = /\{%\s*img\b([\s\S]*?)%\}/g;
const ATTR = /([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
const HAS_HTML = /<[a-zA-Z/]/;

function parseAttrs(raw: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  for (const m of raw.matchAll(ATTR)) attrs[m[1]] = m[2] ?? m[3];
  return attrs;
}

/** Makes an HTML fragment safe to put inside a JSX fragment. */
function toJsx(html: string): string {
  return html
    .replace(/<br\s*>/g, '<br />')
    .replace(/[{}]/g, (c) => (c === '{' ? '&#123;' : '&#125;'));
}

function toAttr(value: string): string {
  return `"${value.replace(/&/g, '&amp;').replace(/"/g, '&quot;')}"`;
}

function toBackground(style: string): string {
  const m = /^\s*background-color:\s*([^;]+?)\s*;?\s*$/.exec(style);
  if (!m) throw new Error(`Unsupported img style: ${style}`);
  return m[1];
}

/** `{% img … %}` becomes `<Figure … />`. The `class` attribute is dropped. */
export function convertImg(body: string): string {
  return outsideCode(body, (text) =>
    text.replace(IMG, (_, raw: string) => {
      const a = parseAttrs(raw);
      if (!a.src) throw new Error(`img tag without src: ${raw}`);
      const props = [`src=${toAttr(a.src)}`];
      if (a.caption) {
        props.push(
          HAS_HTML.test(a.caption)
            ? `caption={<>${toJsx(a.caption)}</>}`
            : `caption=${toAttr(a.caption)}`,
        );
      }
      if (a.align) props.push(`align=${toAttr(a.align)}`);
      if (a.style) props.push(`background=${toAttr(toBackground(a.style))}`);
      return `<Figure ${props.join(' ')} />`;
    }),
  );
}
