import { describe, expect, it } from "vitest";
import { readingTime } from "../src/lib/reading-time";

describe("readingTime", () => {
  it("returns at least 1 minute for empty text", () => {
    expect(readingTime("")).toBe(1);
  });

  it("counts English words at 200 per minute", () => {
    expect(readingTime(Array(400).fill("word").join(" "))).toBe(2);
  });

  it("counts Hangul syllables at 500 per minute", () => {
    expect(readingTime("가".repeat(1000))).toBe(2);
  });

  it("ignores fenced code", () => {
    const code = "```py\n" + Array(2000).fill("x").join(" ") + "\n```";
    expect(readingTime(`hello\n${code}`)).toBe(1);
  });
});
