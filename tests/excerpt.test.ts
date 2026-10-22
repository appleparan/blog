import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { excerpt } from '../src/lib/excerpt';

describe('excerpt', () => {
  it('skips imports and headings and returns the first prose paragraph', () => {
    const body =
      "import Figure from '../x.astro';\n\n## Intro\n\nFirst **bold** paragraph.\n\nSecond.";
    expect(excerpt(body)).toBe('First bold paragraph.');
  });

  it('skips fenced code, math blocks, JSX blocks, and tables', () => {
    const body = [
      '```python',
      'x = 1',
      '```',
      '',
      '$$',
      'a = b',
      '$$',
      '',
      '<Quote cite="a">',
      '',
      'Quoted text.',
      '',
      '</Quote>',
      '',
      '| a | b |',
      '|---|---|',
      '',
      'Real text.',
    ].join('\n');
    expect(excerpt(body)).toBe('Real text.');
  });

  it('strips links, images, inline code, inline math, citations, and Liquid leftovers', () => {
    const body =
      'See [the docs](https://a.b) and ![img](/x.png) with `code`, $x^2$, [@key2019] {% raw %}done{% endraw %}.';
    expect(excerpt(body)).toBe('See the docs and with code, , done.');
  });

  it('cuts to max characters', () => {
    const out = excerpt('가'.repeat(400));
    expect(out.length).toBe(150);
    expect(out.endsWith('…')).toBe(true);
    expect(excerpt('abcdef', 4)).toBe('abc…');
  });

  it('returns an empty string when there is no prose', () => {
    expect(excerpt("import A from 'a';\n\n## Only heading")).toBe('');
  });

  it('works on a real post body that starts with imports', () => {
    const raw = readFileSync(
      'src/content/posts/2020-08-25-External-Forcing-Turbulence.mdx',
      'utf8',
    );
    const body = raw.replace(/^---\n[\s\S]*?\n---\n/, '');
    expect(body.trimStart().startsWith('import')).toBe(true);
    const out = excerpt(body);
    expect(out).toMatch(/^\(This content is originally written by Kyongmin Yeo's manual\)/);
    expect(out).not.toMatch(/[[\]()]\(|import|\*\*/);
    expect(out.length).toBeLessThanOrEqual(150);
  });
});
