import { existsSync, readFileSync } from 'node:fs';
import { parse } from 'node-html-parser';
import { describe, expect, it } from 'vitest';

// Reads the built site, so run `bun run build` before `bun run test`.
const POST = 'dist/posts/2026/10/Inside-vLLM/index.html';

const steppers = () => parse(readFileSync(POST, 'utf8')).querySelectorAll('[data-figure-steps]');

describe('stepped figures', () => {
  it('renders at least one stepped figure in the vLLM post', () => {
    expect(steppers().length).toBeGreaterThan(0);
  });

  it('shows only the first frame before any script runs', () => {
    for (const fig of steppers()) {
      const frames = fig.querySelectorAll('[data-step]');
      expect(frames.length).toBeGreaterThan(1);
      expect(frames.map((f) => f.getAttribute('aria-hidden'))).toEqual(
        frames.map((_, i) => (i === 0 ? 'false' : 'true')),
      );
    }
  });

  it('gives every frame a light and a dark image that exist', () => {
    for (const fig of steppers()) {
      for (const frame of fig.querySelectorAll('[data-step]')) {
        const srcs = frame.querySelectorAll('img').map((img) => img.getAttribute('src')!);
        expect(srcs).toHaveLength(2);
        for (const src of srcs) expect(existsSync(`public${src}`), src).toBe(true);
      }
    }
  });

  it('has one label per frame and buttons to move between frames', () => {
    for (const fig of steppers()) {
      const frames = fig.querySelectorAll('[data-step]');
      const labels = JSON.parse(fig.getAttribute('data-labels')!) as string[];
      expect(labels).toHaveLength(frames.length);
      expect(fig.querySelector('[data-fs-prev]')?.getAttribute('aria-label')).toBeTruthy();
      expect(fig.querySelector('[data-fs-next]')?.getAttribute('aria-label')).toBeTruthy();
      expect(fig.querySelector('[data-fs-label]')?.getAttribute('aria-live')).toBe('polite');
    }
  });
});
