import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { compile } from '@mdx-js/mdx';
import remarkMath from 'remark-math';
import { describe, expect, it } from 'vitest';
import {
  convertCite,
  convertFrontmatter,
  convertImg,
  convertPost,
  convertQuote,
  convertYoutube,
  escapeMdx,
  removeBibliography,
} from '../src/lib/convert';

describe('convertCite', () => {
  it('converts a cite and collects the file', () => {
    expect(
      convertCite('(cf. {% cite Orszag1971 --file 2017-07-17-Psuedo-spectral-method %}, x)'),
    ).toEqual({ body: '(cf. [@Orszag1971], x)', files: ['2017-07-17-Psuedo-spectral-method'] });
  });

  it('tolerates irregular spacing', () => {
    expect(convertCite('{% cite Yu1992  --file F  %}').body).toBe('[@Yu1992]');
  });

  it('joins several keys', () => {
    expect(convertCite('{% cite a b --file F %}').body).toBe('[@a; @b]');
  });

  it('leaves fenced code alone', () => {
    const code = '```md\n{% cite x --file F %}\n```';
    expect(convertCite(code)).toEqual({ body: code, files: [] });
  });
});

describe('removeBibliography', () => {
  it('drops the tag and a bare References heading', () => {
    expect(
      removeBibliography('text\n\n## References\n\n{% bibliography --cited --file F %}\n'),
    ).toBe('text\n\n');
  });

  it('keeps other headings', () => {
    expect(removeBibliography('## Notes\n\n{% bibliography --file F %}\n')).toBe('## Notes\n\n');
  });
});

describe('escapeMdx', () => {
  it('escapes braces outside code and math', () => {
    expect(escapeMdx('set {a} and $$\\mathbf{s}_\\phi$$ and `{x}`')).toBe(
      'set \\{a\\} and $$\\mathbf{s}_\\phi$$ and `{x}`',
    );
  });

  it('escapes unknown tags, keeps known ones, closes br', () => {
    expect(escapeMdx('<harbor_URL> <a href="x">y</a> a<br>b')).toBe(
      '&lt;harbor_URL> <a href="x">y</a> a<br />b',
    );
  });

  it('turns kramdown \\\\( \\\\) math into dollar math', () => {
    expect(escapeMdx('where \\\\(a_{i} < b\\\\) holds')).toBe('where $a_{i} < b$ holds');
  });

  it('keeps display blocks and rewrites line-start inline $$ pairs', () => {
    expect(escapeMdx('$$\nx_{1}\n$$\n')).toBe('$$\nx_{1}\n$$\n');
    expect(escapeMdx('$$a_{1},\nb$$ then text')).toBe('$a_{1},\nb$ then text');
    expect(escapeMdx('text $$a_{1}$$ more')).toBe('text $$a_{1}$$ more');
  });

  it('leaves fences, Liquid tags and single-dollar math untouched', () => {
    const src =
      "```sh\necho $HF_HOME {% cite x %} <y>\n```\n\n{% img caption='<b>{</b>' %} $a_{1}$";
    expect(escapeMdx(src)).toBe(src);
  });
});

describe('convertImg', () => {
  it('uses a JSX caption for HTML captions', () => {
    const out = convertImg(
      `{% img align="center" style='background-color: #fff' caption='<a href="https://x.y">Batch</a>' src='/assets/a.png' %}`,
    );
    expect(out).toBe(
      '<Figure src="/assets/a.png" caption={<><a href="https://x.y">Batch</a></>} align="center" background="#fff" />',
    );
  });

  it('uses a string caption for plain text and handles multi-line tags', () => {
    expect(convertImg(`{% img align="center"\n caption='Plain "x"' src='/a.png' %}`)).toBe(
      '<Figure src="/a.png" caption="Plain &quot;x&quot;" align="center" />',
    );
  });

  it('rejects unknown styles', () => {
    expect(() => convertImg(`{% img style='color: red' src='/a.png' %}`)).toThrow(/style/);
  });
});

describe('convertQuote', () => {
  it('wraps the inside and collects the file', () => {
    expect(convertQuote('{% quote k1 --file F %}\nHello\n{% endquote %} \n')).toEqual({
      body: '<Quote cite="k1">\n\nHello\n\n</Quote>\n',
      files: ['F'],
    });
  });
});

describe('convertYoutube', () => {
  it('converts to the YouTube component', () => {
    expect(convertYoutube('{% youtube "https://www.youtube.com/watch?v=FlR8CvyaE4I" %}')).toBe(
      '<YouTube id="https://www.youtube.com/watch?v=FlR8CvyaE4I" />',
    );
  });
});

describe('convertFrontmatter', () => {
  it('adds bibliography and imports', () => {
    expect(convertFrontmatter('title: T\ndraft: true', 'F', ['Figure', 'YouTube'])).toBe(
      "---\ntitle: T\ndraft: true\nbibliography: F.bib\n---\n\nimport Figure from '../../components/Figure.astro';\nimport { YouTube } from '@astro-community/astro-embed-youtube';\n",
    );
  });
});

describe('convertPost', () => {
  it('keeps fenced code intact and converts the rest', () => {
    const src = [
      '---',
      'title: T',
      '---',
      '',
      'See {% cite a --file F %} and {x}.',
      '',
      '```sh',
      'echo $HF_HOME {% cite x %}',
      '```',
      '',
    ].join('\n');
    const out = convertPost(src, '2020-01-01-T.md');
    expect(out).toContain('bibliography: F.bib');
    expect(out).toContain('See [@a] and \\{x\\}.');
    expect(out).toContain('```sh\necho $HF_HOME {% cite x %}\n```');
  });

  it('refuses posts that cite two bibliography files', () => {
    const src = '---\ntitle: T\n---\n{% cite a --file F %} {% cite b --file G %}\n';
    expect(() => convertPost(src, 'x.md')).toThrow(/more than one/);
  });
});

// Task 5 deletes _posts/, so this check skips itself once the Jekyll sources are gone.
describe.skipIf(!existsSync('_posts'))('real posts', () => {
  const files = existsSync('_posts') ? readdirSync('_posts').filter((f) => f.endsWith('.md')) : [];

  it.each(files)('%s converts and compiles as MDX', async (file) => {
    const mdx = convertPost(readFileSync(join('_posts', file), 'utf8'), file);
    expect(mdx).not.toMatch(/\{%/);
    await expect(compile(mdx, { remarkPlugins: [remarkMath] })).resolves.toBeDefined();
  });
});
