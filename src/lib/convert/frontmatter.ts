const IMPORTS: Record<string, string> = {
  Figure: "import Figure from '../../components/Figure.astro';",
  Quote: "import Quote from '../../components/Quote.astro';",
  YouTube: "import { YouTube } from '@astro-community/astro-embed-youtube';",
};

/** Components used in a converted body, in a stable order. */
export function usedComponents(body: string): string[] {
  return Object.keys(IMPORTS).filter((name) => new RegExp(`<${name}[\\s/>]`).test(body));
}

/**
 * Keeps every front matter field as written except `layout` (MDX would import it as a layout
 * module), adds `bibliography: <F>.bib` when the post cites, and returns the front matter block
 * followed by the MDX imports.
 */
export function convertFrontmatter(
  fm: string,
  bibFile: string | null,
  components: string[] = [],
): string {
  const lines = [fm.replace(/^layout:.*\n?/m, '').replace(/\s+$/, '')];
  if (bibFile) lines.push(`bibliography: ${bibFile}.bib`);
  const header = `---\n${lines.join('\n')}\n---\n`;
  if (components.length === 0) return header;
  return `${header}\n${components.map((c) => IMPORTS[c]).join('\n')}\n`;
}
