export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface PopoverPosition {
  left: number;
  top: number;
  placement: 'above' | 'below';
}

/** Space kept between the popover and the viewport edge, in px. */
export const VIEWPORT_MARGIN = 8;
/** Space between the link and the popover, in px. */
export const LINK_GAP = 6;

/**
 * Places a popover next to a link, in viewport coordinates.
 * It goes below the link and flips above when the space below is too small and the space
 * above is larger. It is then clamped inside the viewport with an 8px margin.
 */
export function computePopoverPosition(link: Rect, pop: Size, viewport: Size): PopoverPosition {
  const spaceBelow = viewport.height - link.bottom - LINK_GAP - VIEWPORT_MARGIN;
  const spaceAbove = link.top - LINK_GAP - VIEWPORT_MARGIN;
  const placement = pop.height > spaceBelow && spaceAbove > spaceBelow ? 'above' : 'below';
  const top = placement === 'below' ? link.bottom + LINK_GAP : link.top - LINK_GAP - pop.height;
  const maxLeft = viewport.width - VIEWPORT_MARGIN - pop.width;
  const left = Math.max(VIEWPORT_MARGIN, Math.min(link.left, maxLeft));
  return { left, top, placement };
}
