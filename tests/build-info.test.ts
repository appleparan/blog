import { describe, expect, it } from "vitest";
import { formatBuildDate, formatVersion } from "../src/lib/build-info";

describe("formatVersion", () => {
  it("returns the tag when the commit is tagged", () => {
    expect(formatVersion("v2.0.0")).toBe("v2.0.0");
  });

  it("appends the commit count after the latest tag", () => {
    expect(formatVersion("v2.0.0-3-gabc1234")).toBe("v2.0.0+3");
  });

  it("falls back to dev when git describe gives nothing", () => {
    expect(formatVersion("")).toBe("dev");
  });
});

describe("formatBuildDate", () => {
  it("formats the date in Asia/Seoul as YYYY-MM-DD", () => {
    // 2026-10-03 16:00 UTC is 2026-10-04 01:00 in Seoul.
    expect(formatBuildDate(new Date("2026-10-03T16:00:00Z"))).toBe("2026-10-04");
  });
});
