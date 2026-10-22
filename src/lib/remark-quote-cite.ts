/** The mdast fields this plugin touches; MDX JSX nodes come from mdast-util-mdx-jsx. */
interface Node {
  type: string;
  name?: string | null;
  attributes?: { type: string; name?: string; value?: unknown }[];
  children?: Node[];
  data?: Record<string, unknown>;
  value?: string;
}

/** Key of `<Quote cite="key">`, or undefined for any other node. */
function quoteCite(node: Node): string | undefined {
  if (node.type !== 'mdxJsxFlowElement' || node.name !== 'Quote') return undefined;
  const attr = node.attributes?.find((a) => a.type === 'mdxJsxAttribute' && a.name === 'cite');
  return typeof attr?.value === 'string' ? attr.value : undefined;
}

/**
 * Appends `[@key]` from `<Quote cite="key">` as a `slot="cite"` paragraph, so
 * rehype-citation turns it into a numbered link and Quote.astro shows it in the footer.
 */
export function remarkQuoteCite() {
  const walk = (node: Node): void => {
    node.children?.forEach(walk);
    const cite = quoteCite(node);
    if (cite === undefined) return;
    node.children = [
      ...(node.children ?? []),
      {
        type: 'paragraph',
        data: { hProperties: { slot: 'cite' } },
        children: [{ type: 'text', value: `[@${cite}]` }],
      },
    ];
  };
  return walk;
}
