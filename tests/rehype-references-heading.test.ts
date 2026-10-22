import { describe, expect, it } from 'vitest';
import { rehypeReferencesHeading } from '../src/lib/rehype-references-heading';

const refs = () => ({ type: 'element', tagName: 'div', properties: { id: 'refs' }, children: [] });
const p = () => ({ type: 'element', tagName: 'p', properties: {}, children: [] });

describe('rehypeReferencesHeading', () => {
  it('puts a References heading right before the bibliography', () => {
    const tree = { type: 'root', children: [p(), refs()] };
    rehypeReferencesHeading()(tree);
    expect(tree.children.map((c) => c.tagName)).toEqual(['p', 'h2', 'div']);
    expect(tree.children[1]).toMatchObject({
      properties: { id: 'references' },
      children: [{ type: 'text', value: 'References' }],
    });
  });

  it('leaves a post without a bibliography alone', () => {
    const tree = { type: 'root', children: [p()] };
    rehypeReferencesHeading()(tree);
    expect(tree.children).toHaveLength(1);
  });
});
