import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
// CommonMark does not close `**` between punctuation and a letter, so Korean like
// `**일반화(generalization)**라는` stays as literal asterisks without remark-cjk-friendly.
const POST = 'dist/posts/2025/10/Preventing-Overfitting-With-Norms-and-BN/index.html';

describe('CJK emphasis', () => {
  it('renders bold that ends with punctuation and is followed by Hangul', () => {
    const html = readFileSync(POST, 'utf8');
    expect(html).toContain('<strong>일반화(generalization)</strong>라는');
    expect(html).not.toContain('**일반화(generalization)**');
  });
});
