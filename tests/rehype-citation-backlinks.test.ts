// @ts-nocheck -- hand-built hast trees; the plugin takes a loose Node type.
import { describe, expect, it } from 'vitest';
import { rehypeCitationBacklinks } from '../src/lib/rehype-citation-backlinks';

const text = (value) => ({ type: 'text', value });
const el = (tagName, properties = {}, children = []) => ({
  type: 'element',
  tagName,
  properties,
  children,
});
const cite = (key, n, label = '1') =>
  el('span', { id: `citation--${key}--${n}` }, [
    text('['),
    el('a', { href: `#bib-${key}` }, [text(label)]),
    text(']'),
  ]);
const entry = (key) =>
  el('div', { className: ['csl-entry'], id: `bib-${key}` }, [
    el('div', { className: ['csl-left-margin'] }, [text('[1]')]),
    el('div', { className: ['csl-right-inline'] }, [text('Some paper')]),
  ]);
const refs = (...keys) => el('div', { id: 'refs', className: ['references'] }, keys.map(entry));
const backlinks = (e) =>
  e.children.filter((c) => c.tagName === 'a' && c.properties.className?.includes('backlink'));

describe('rehypeCitationBacklinks', () => {
  it('adds one plain back-link for a key cited once', () => {
    const tree = { type: 'root', children: [el('p', {}, [cite('a', 1)]), refs('a')] };
    rehypeCitationBacklinks()(tree);
    const links = backlinks(tree.children[1].children[0]);
    expect(links).toHaveLength(1);
    expect(links[0].properties).toMatchObject({
      href: '#citation--a--1',
      ariaLabel: '본문으로 돌아가기',
    });
    expect(links[0].children).toEqual([text('↩')]);
  });

  it('labels several back-links a, b, c in text order', () => {
    const tree = {
      type: 'root',
      children: [
        el('p', {}, [cite('a', 1), cite('b', 2), cite('a', 3), cite('a', 4)]),
        refs('a', 'b'),
      ],
    };
    rehypeCitationBacklinks()(tree);
    const [a, b] = tree.children[1].children;
    expect(backlinks(a).map((l) => [l.properties.href, l.children[0].value])).toEqual([
      ['#citation--a--1', '↩ a'],
      ['#citation--a--3', '↩ b'],
      ['#citation--a--4', '↩ c'],
    ]);
    expect(backlinks(b).map((l) => l.children[0].value)).toEqual(['↩']);
  });

  it('gives each key of a multi-link citation span a back-link to that span', () => {
    const span = el('span', { id: 'citation--a--1' }, [
      text('['),
      el('a', { href: '#bib-a' }, [text('1')]),
      text(', '),
      el('a', { href: '#bib-b' }, [text('2')]),
      text(']'),
    ]);
    const tree = { type: 'root', children: [el('p', {}, [span]), refs('a', 'b')] };
    rehypeCitationBacklinks()(tree);
    const [a, b] = tree.children[1].children;
    expect(backlinks(a)[0].properties.href).toBe('#citation--a--1');
    expect(backlinks(b)[0].properties.href).toBe('#citation--a--1');
  });

  it('adds no back-link to an entry nobody cites', () => {
    const tree = { type: 'root', children: [el('p', {}, [cite('a', 1)]), refs('a', 'unused')] };
    rehypeCitationBacklinks()(tree);
    expect(backlinks(tree.children[1].children[1])).toHaveLength(0);
  });

  it('leaves a tree without a bibliography alone', () => {
    const tree = { type: 'root', children: [el('p', {}, [cite('a', 1)])] };
    rehypeCitationBacklinks()(tree);
    expect(tree.children).toHaveLength(1);
    expect(tree.children[0].children).toHaveLength(1);
  });
});
