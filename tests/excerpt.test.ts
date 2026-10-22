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

  it('strips links, images, inline code, citations, and Liquid leftovers', () => {
    const body =
      'See [the docs](https://a.b) and ![img](/x.png) with `code`, $x^2$, [@key2019] {% raw %}done{% endraw %}.';
    expect(excerpt(body)).toBe('See the docs and with code, x^2, done.');
  });

  it('keeps inline math as TeX source and leaves Markdown rules out of it', () => {
    expect(excerpt('Let $x_i < y_i$ and $a * b * c$ hold.')).toBe(
      'Let x_i < y_i and a * b * c hold.',
    );
  });

  it('keeps inline math inside parentheses (Logit post)', () => {
    const body =
      '도박을 생각해보자. odds란 도박의 승률을 나타내는 중요한 지표이다. 확률이 이길 확률($p$)과, 전체 경우의 수(이김 + 짐)의 합(1)으로 $\\dfrac{p}{(p + (1-p))} = \\dfrac{p}{1} = p$ 표시되는 형태라면, odds는 이길 확률($p$)과 질 확률($1-p$)의 비율로 표시된다.';
    const out = excerpt(body, 300);
    expect(out).toContain('이길 확률(p)과');
    expect(out).toContain('질 확률(1-p)의');
    expect(out).not.toContain('()');
  });

  it('drops the sentence a citation is the subject of (SVD post)', () => {
    const body = [
      'SVD(Singular Value Decomposition)는 LLM을 공부하다보면 참 이곳저곳에서 많이 나오는 알고리즘이다.',
      '특히 [@strang2019linear]에서도 **Data Science meets linear algebra in the SVD**라고 시작하고,',
      '[Low Level Technicals of LLMs: Daniel Han](https://youtu.be/pRM_P6UfdIc?t=411)에서도 그 중요성을 강조하고 있다.',
    ].join('\n');
    expect(excerpt(body)).toBe(
      'SVD(Singular Value Decomposition)는 LLM을 공부하다보면 참 이곳저곳에서 많이 나오는 알고리즘이다.',
    );
  });

  it('drops a cited sentence in the middle of a paragraph (Attention post)', () => {
    const body = [
      '기존의 SDPA(Scaled Dot Product Attention)를 효율화하는 여러가지 방법이 있다.',
      '대표적으로 [@vaswani2017attention] 에도 나오는',
      'MHA(Multi-Head Attention)부터 시작해서, MQA(Multi-Query Attention), GQA(Grouped-Query Attention), 그리고 MLA(Multi-head Latent Header Attention)에 대해 알아보고 KV Cache가 얼마나 optimize되는지 알아보고자 한다.',
    ].join('\n');
    expect(excerpt(body)).toBe(
      '기존의 SDPA(Scaled Dot Product Attention)를 효율화하는 여러가지 방법이 있다.',
    );
  });

  it('removes citations and empty parentheses when every sentence cites', () => {
    expect(excerpt('Proposed by Smith ([@smith2020]) in 2020 [@a; @b].')).toBe(
      'Proposed by Smith in 2020.',
    );
  });

  it('reads a paragraph that sits directly under its heading', () => {
    expect(excerpt('## Introduction\n개인적으로 정리하는 포스트이다.\n\n## Next')).toBe(
      '개인적으로 정리하는 포스트이다.',
    );
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
