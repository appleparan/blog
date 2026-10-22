const HANGUL_PER_MINUTE = 500;
const WORDS_PER_MINUTE = 200;

/** Estimated reading time in minutes (at least 1). Fenced code is ignored. */
export function readingTime(text: string): number {
  const prose = text.replace(/^(```|~~~)[\s\S]*?^\1[^\n]*$/gm, " ");
  const hangul = prose.match(/[가-힣]/g)?.length ?? 0;
  const words = prose
    .replace(/[가-힣]/g, " ")
    .split(/\s+/)
    .filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
  const minutes = hangul / HANGUL_PER_MINUTE + words / WORDS_PER_MINUTE;
  return Math.max(1, Math.ceil(minutes));
}
