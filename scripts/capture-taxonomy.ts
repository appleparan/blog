// Usage: bun scripts/capture-taxonomy.ts <postsDir> <outFile>
// Freezes the category and tag names of published Jekyll posts, so tests survive the removal of _posts/.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [postsDir, outFile] = process.argv.slice(2);
if (!postsDir || !outFile) throw new Error('usage: capture-taxonomy.ts <postsDir> <outFile>');

const categories = new Set<string>();
const tags = new Set<string>();
for (const file of readdirSync(postsDir).filter((f) => /^\d{4}-\d{2}-\d{2}-.+\.md$/.test(f))) {
  const front = Bun.YAML.parse(readFileSync(join(postsDir, file), 'utf8').split(/^---\s*$/m)[1]) as {
    draft?: boolean;
    categories?: string[];
    tags?: unknown[];
  };
  if (front.draft === true) continue;
  for (const c of front.categories ?? []) categories.add(String(c));
  for (const t of front.tags ?? []) tags.add(String(t));
}
const sort = (s: Set<string>): string[] => [...s].sort();
writeFileSync(outFile, JSON.stringify({ categories: sort(categories), tags: sort(tags) }, null, 2) + '\n');
