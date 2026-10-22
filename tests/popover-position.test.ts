import { describe, expect, it } from 'vitest';
import { computePopoverPosition } from '../src/lib/popover-position';

const viewport = { width: 1000, height: 800 };
const rect = (left: number, top: number, width = 20, height = 20) => ({
  left,
  top,
  right: left + width,
  bottom: top + height,
  width,
  height,
});

describe('computePopoverPosition', () => {
  it('places the popover below the link, left-aligned', () => {
    const p = computePopoverPosition(rect(300, 100), { width: 400, height: 150 }, viewport);
    expect(p).toMatchObject({ left: 300, top: 126, placement: 'below' });
  });

  it('flips above when there is no room below', () => {
    const p = computePopoverPosition(rect(300, 700), { width: 400, height: 150 }, viewport);
    expect(p.placement).toBe('above');
    expect(p.top).toBe(700 - 6 - 150);
  });

  it('stays below when neither side fits but below has more room', () => {
    const p = computePopoverPosition(rect(300, 300), { width: 400, height: 700 }, viewport);
    expect(p.placement).toBe('below');
  });

  it('clamps to the right edge with an 8px margin', () => {
    const p = computePopoverPosition(rect(900, 100), { width: 400, height: 150 }, viewport);
    expect(p.left).toBe(1000 - 8 - 400);
  });

  it('clamps to the left edge with an 8px margin', () => {
    const p = computePopoverPosition(rect(-50, 100), { width: 400, height: 150 }, viewport);
    expect(p.left).toBe(8);
  });

  it('pins to the left margin when the popover is wider than the viewport', () => {
    const p = computePopoverPosition(
      rect(100, 100),
      { width: 500, height: 150 },
      { width: 390, height: 800 },
    );
    expect(p.left).toBe(8);
  });
});
