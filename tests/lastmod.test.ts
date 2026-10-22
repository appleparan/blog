import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { lastModified } from '../src/lib/lastmod';

// A shallow clone (e.g. a CI checkout without fetch-depth: 0) lacks the old history.
const shallow =
  execFileSync('git', ['rev-parse', '--is-shallow-repository'], { encoding: 'utf8' }).trim() ===
  'true';

describe('lastModified', () => {
  it('returns null for a file outside a git repository', () => {
    const file = join(mkdtempSync(join(tmpdir(), 'lastmod-')), 'post.mdx');
    writeFileSync(file, 'x');
    expect(lastModified(file)).toBeNull();
  });

  it('returns null for a missing file', () => {
    expect(lastModified('src/content/posts/does-not-exist.mdx')).toBeNull();
  });

  it.skipIf(shallow)('follows a post back through the move out of _posts/', () => {
    // The move and the conversion commits carry `Lastmod: skip`; the post's last real edit
    // is "Fix \"Reference\" to plural form", authored 2024-05-18T23:29:15+09:00.
    expect(lastModified('src/content/posts/2018-07-14-Tox-Workflow.mdx')?.toISOString()).toBe(
      '2024-05-18T14:29:15.000Z',
    );
  });
});
