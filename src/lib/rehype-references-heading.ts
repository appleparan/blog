/** The hast fields this plugin touches. */
interface Node {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: Node[];
  value?: string;
}

const HEADING = /^h[1-6]$/;
const REFERENCES_TEXT = /^(?:references?|참고문헌)$/i;

const isBibliography = (node: Node): boolean =>
  node.type === 'element' && node.properties?.id === 'refs';

const textOf = (node: Node): string =>
  node.type === 'text' ? (node.value ?? '') : (node.children ?? []).map(textOf).join('');

/** True when the tree already has a References heading (by text or by id). */
function hasReferencesHeading(node: Node): boolean {
  if (node.type === 'element' && HEADING.test(node.tagName ?? '')) {
    if (node.properties?.id === 'references' || REFERENCES_TEXT.test(textOf(node).trim())) {
      return true;
    }
  }
  return (node.children ?? []).some(hasReferencesHeading);
}

/**
 * Adds `<h2 id="references">References</h2>` before rehype-citation's `<div id="refs">`, which
 * the plugin appends at the end of the content. The converter drops a Jekyll `## References`
 * heading only when `{% bibliography %}` follows it directly; when a link list sits between them,
 * the author's heading stays, and this plugin adds nothing (no second heading, no duplicate id).
 * Astro lists the added heading in the post's headings for the TOC. Run it after rehype-citation.
 */
export function rehypeReferencesHeading() {
  const insert = (node: Node): boolean => {
    const children = node.children ?? [];
    const i = children.findIndex(isBibliography);
    if (i < 0) return children.some(insert);
    const heading: Node = {
      type: 'element',
      tagName: 'h2',
      properties: { id: 'references' },
      children: [{ type: 'text', value: 'References' }],
    };
    node.children = [...children.slice(0, i), heading, ...children.slice(i)];
    return true;
  };
  return (tree: Node): void => {
    if (!hasReferencesHeading(tree)) insert(tree);
  };
}
