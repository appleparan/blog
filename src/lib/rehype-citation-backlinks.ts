/** The hast fields this plugin touches. */
interface Node {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: Node[];
  value?: string;
}

const BACKLINK_LABEL = '본문으로 돌아가기';
const CITATION_ID = /^citation--/;
const BIB_HREF = /^#bib-(.+)$/;

const classOf = (node: Node): string[] => {
  const c = node.properties?.className;
  return Array.isArray(c) ? (c as string[]) : typeof c === 'string' ? c.split(/\s+/) : [];
};

const isElement = (node: Node): boolean => node.type === 'element';

/** Walks the tree in document order and calls `visit` on every node. */
function walk(node: Node, visit: (n: Node) => void): void {
  visit(node);
  for (const child of node.children ?? []) walk(child, visit);
}

/** Maps each bibliography key to the ids of the citation spans that cite it, in text order. */
function collectCitations(tree: Node): Map<string, string[]> {
  const byKey = new Map<string, string[]>();
  walk(tree, (node) => {
    const id = node.properties?.id;
    if (!isElement(node) || typeof id !== 'string' || !CITATION_ID.test(id)) return;
    const keys = new Set<string>();
    walk(node, (n) => {
      const href = n.tagName === 'a' ? n.properties?.href : undefined;
      const key = typeof href === 'string' ? BIB_HREF.exec(href)?.[1] : undefined;
      if (key) keys.add(key);
    });
    for (const key of keys) byKey.set(key, [...(byKey.get(key) ?? []), id]);
  });
  return byKey;
}

const backlink = (href: string, label: string): Node => ({
  type: 'element',
  tagName: 'a',
  properties: { href, className: ['backlink'], ariaLabel: BACKLINK_LABEL },
  children: [{ type: 'text', value: label }],
});

/** `↩` for one citation; `↩ a`, `↩ b`, ... for several. */
const labelFor = (index: number, total: number): string =>
  total === 1 ? '↩' : `↩ ${String.fromCharCode(97 + index)}`;

/**
 * Appends back-links to each bibliography entry, one per in-text citation of its key. Each link
 * points to the citation span (`#citation--<key>--<n>`). Entries nobody cites (`noCite`) get none.
 * Run it after rehype-citation, which writes both the spans and the entries.
 */
export function rehypeCitationBacklinks() {
  return (tree: Node): void => {
    const citations = collectCitations(tree);
    walk(tree, (node) => {
      const id = node.properties?.id;
      if (!isElement(node) || !classOf(node).includes('csl-entry') || typeof id !== 'string') {
        return;
      }
      const ids = citations.get(id.replace(/^bib-/, '')) ?? [];
      if (ids.length === 0) return;
      const links = ids.map((spanId, i) => backlink(`#${spanId}`, labelFor(i, ids.length)));
      node.children = [...(node.children ?? []), ...links];
    });
  };
}
