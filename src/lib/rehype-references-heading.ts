/** The hast fields this plugin touches. */
interface Node {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: Node[];
  value?: string;
}

const isBibliography = (node: Node): boolean =>
  node.type === 'element' && node.properties?.id === 'refs';

/**
 * Adds `<h2 id="references">References</h2>` before rehype-citation's `<div id="refs">`.
 * The converter drops the Jekyll `## References` heading together with `{% bibliography %}`.
 * The heading must sit inside the MDX output, right above the list (the list is the last
 * child of the content), and Astro then lists it in the post's headings for the TOC.
 * Run it after rehype-citation.
 */
export function rehypeReferencesHeading() {
  const walk = (node: Node): void => {
    const children = node.children ?? [];
    const i = children.findIndex(isBibliography);
    if (i >= 0) {
      const heading: Node = {
        type: 'element',
        tagName: 'h2',
        properties: { id: 'references' },
        children: [{ type: 'text', value: 'References' }],
      };
      node.children = [...children.slice(0, i), heading, ...children.slice(i)];
      return;
    }
    children.forEach(walk);
  };
  return (tree: Node): void => walk(tree);
}
