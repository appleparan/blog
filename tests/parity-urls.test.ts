import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
const reference = JSON.parse(readFileSync('tests/fixtures/jekyll-reference.json', 'utf8')) as {
  urls: string[];
};

// Jekyll URLs the Astro site drops on purpose, each with the reason.
const IGNORED: Record<string, string> = {};

// Files the fixture does not list but every reader-facing site must keep.
const REQUIRED_FILES = ['/404.html', '/robots.txt'];

function builtPath(url: string): string {
  return url.endsWith('/') ? `dist${url}index.html` : `dist${url}`;
}

describe('every Jekyll URL exists in the Astro build', () => {
  const urls = reference.urls.filter((url) => !(url in IGNORED));

  it.each(urls)('%s', (url) => {
    expect(existsSync(builtPath(url))).toBe(true);
  });

  it.each(REQUIRED_FILES)('%s', (file) => {
    expect(existsSync(`dist${file}`)).toBe(true);
  });

  it('ignores only URLs that are in the fixture', () => {
    expect(Object.keys(IGNORED).filter((url) => !reference.urls.includes(url))).toEqual([]);
  });
});
