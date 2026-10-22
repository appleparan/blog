import { readdirSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
describe('canonical URLs', () => {
  const htmlFiles = readdirSync('dist', { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('.html'))
    .sort();
  const head = (file: string) => parse(readFileSync(`dist/${file}`, 'utf8'));
  // `dist/a/b/index.html` is served at `/a/b/`; non-ASCII paths are percent-encoded.
  const ownUrl = (file: string) =>
    new URL(file.replace(/(^|\/)index\.html$/, '$1'), 'https://blog.liam.kim/').href;

  it.each(htmlFiles.filter((f) => f !== '404.html'))(
    '%s points canonical and og:url at itself',
    (file) => {
      const root = head(file);
      expect(root.querySelectorAll('link[rel=canonical]')).toHaveLength(1);
      expect(root.querySelector('link[rel=canonical]')?.getAttribute('href')).toBe(ownUrl(file));
      expect(root.querySelector("meta[property='og:url']")?.getAttribute('content')).toBe(
        ownUrl(file),
      );
      expect(root.querySelector('meta[name=robots]')).toBeNull();
    },
  );

  it('builds more than the home page', () => {
    expect(htmlFiles).toContain('categories/vision/index.html');
    expect(htmlFiles).toContain('tags/diffusion-models/index.html');
  });

  it('marks the 404 page noindex with no canonical or og:url', () => {
    const root = head('404.html');
    expect(root.querySelector('meta[name=robots]')?.getAttribute('content')).toBe('noindex');
    expect(root.querySelector('link[rel=canonical]')).toBeNull();
    expect(root.querySelector("meta[property='og:url']")).toBeNull();
  });

  it('sets og:locale to ko_KR', () => {
    expect(
      head('index.html').querySelector("meta[property='og:locale']")?.getAttribute('content'),
    ).toBe('ko_KR');
  });
});
