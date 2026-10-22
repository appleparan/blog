// Usage: bun scripts/convert-posts.ts <postsDir> <outDir>
// Converts Jekyll posts (kramdown + Liquid) to MDX files with the same base name.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { convertPost } from '../src/lib/convert';

const [postsDir, outDir] = process.argv.slice(2);
if (!postsDir || !outDir) throw new Error('usage: convert-posts.ts <postsDir> <outDir>');

mkdirSync(outDir, { recursive: true });
for (const file of readdirSync(postsDir).filter((f) => f.endsWith('.md'))) {
  const mdx = convertPost(readFileSync(join(postsDir, file), 'utf8'), file);
  writeFileSync(join(outDir, file.replace(/\.md$/, '.mdx')), mdx);
  console.log(`converted ${file}`);
}
