import { describe, expect, it } from 'vitest';
import { tocHeadings } from '../src/lib/toc-headings';

// What Astro collects for a heading with KaTeX: MathML text, TeX annotation, and HTML glyphs.
const junk = { depth: 3, slug: 'range-of-ωijomega_ijωij', text: 'Range of ωij\\omega_{ij}ωij​' };

describe('tocHeadings', () => {
  it('renders math in headings from the source TeX', () => {
    const body =
      '## Intro\n\n```sh\n# not a heading $HOME\n```\n\n### Range of $\\omega_{ij}$ <x>\n';
    const [intro, range] = tocHeadings(
      [
        { depth: 2, slug: 'intro', text: 'Intro' },
        { ...junk, text: junk.text + ' <x>' },
      ],
      body,
    );
    expect(intro).toEqual({ depth: 2, slug: 'intro', text: 'Intro' });
    expect(range.slug).toBe(junk.slug);
    expect(range.text).toBe('Range of \\omega_{ij} <x>');
    expect(range.html).toMatch(/^Range of <span class="katex">.*<\/span> &lt;x&gt;$/);
  });

  it('keeps the References heading the plugin adds after the body', () => {
    const refs = { depth: 2, slug: 'references', text: 'References' };
    const out = tocHeadings([junk, refs], '### Range of $\\omega_{ij}$\n');
    expect(out[1]).toEqual(refs);
    expect(out[0].html).toContain('katex');
  });

  it('leaves the headings alone when the source does not line up', () => {
    const headings = [junk];
    expect(tocHeadings(headings, '## Other $x$\n### Range of $\\omega_{ij}$\n')).toBe(headings);
  });
});
