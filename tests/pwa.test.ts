import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { siteConfig } from '../src/config';

// Reads the built site, so run `bun run build` before `bun run test`.
const walk = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)],
  );

describe('service worker', () => {
  it('exists at /sw.min.js with a real build id and the lifecycle calls', () => {
    expect(existsSync('dist/sw.min.js')).toBe(true);
    const sw = readFileSync('dist/sw.min.js', 'utf8');
    expect(sw).not.toContain('__BUILD_ID__');
    expect(sw).toContain('skipWaiting');
    expect(sw).toContain('clients.claim');
    const copies = readdirSync('dist').filter((f) => f.startsWith('sw.min.js'));
    for (const f of copies) {
      const raw = readFileSync(join('dist', f));
      const text = f.endsWith('.gz') ? gunzipSync(raw).toString() : raw.toString();
      expect(text, f).not.toContain('__BUILD_ID__');
    }
    expect(sw).toMatch(/const CACHE = "blog-[\w.-]+"/);
  });
});

describe('manifest', () => {
  const manifests = ['dist/manifest.webmanifest', 'dist/assets/img/favicons/site.webmanifest'];

  it.each(manifests)('%s is valid JSON with the site name and standalone display', (path) => {
    const m = JSON.parse(readFileSync(path, 'utf8'));
    expect(m.name).toBe(siteConfig.title);
    expect(m.display).toBe('standalone');
    expect(m.theme_color).toBe('#fafaf8');
    for (const icon of m.icons) {
      expect(existsSync(`dist${icon.src}`)).toBe(true);
    }
  });

  it('both manifest URLs serve the same content', () => {
    expect(readFileSync(manifests[1], 'utf8')).toBe(readFileSync(manifests[0], 'utf8'));
  });

  it('is linked from every page', () => {
    const pages = walk('dist').filter((f) => f.endsWith('.html'));
    expect(pages.length).toBeGreaterThan(20);
    const missing = pages.filter(
      (f) => !readFileSync(f, 'utf8').includes('rel="manifest" href="/manifest.webmanifest"'),
    );
    expect(missing).toEqual([]);
  });

  it('registers the worker from the built pages', () => {
    const home = readFileSync('dist/index.html', 'utf8');
    expect(home).toContain("serviceWorker.register('/sw.min.js'");
  });
});

describe('static assets', () => {
  it('leave no Liquid tags under dist/assets', () => {
    const bad = walk('dist/assets').filter((f) => {
      if (/\.(png|jpe?g|gif|webp|avif|ico|woff2?|pdf|mp4)$/i.test(f)) return false;
      const text = readFileSync(f, 'utf8');
      return text.includes('{{') || text.includes('{%');
    });
    expect(bad).toEqual([]);
  });

  it('browserconfig.xml is plain XML', () => {
    const xml = readFileSync('dist/assets/img/favicons/browserconfig.xml', 'utf8');
    expect(xml.startsWith('<?xml')).toBe(true);
  });
});
