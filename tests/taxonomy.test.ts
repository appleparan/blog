import { existsSync, readdirSync, readFileSync } from "node:fs";
import { parse } from "node-html-parser";
import { describe, expect, it } from "vitest";

// Reads the built site, so run `bun run build` before `bun run test`.
const fixture = JSON.parse(readFileSync("tests/fixtures/jekyll-reference.json", "utf8")) as {
  urls: string[];
};
const taxonomyUrls = fixture.urls.filter((u) => /^\/(categories|tags)\/[^/]+\/$/.test(u));
const page = (path: string) => parse(readFileSync(`dist${path}index.html`, "utf8"));
const subdirs = (dir: string) =>
  readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name);

describe("taxonomy pages", () => {
  it("builds every category and tag URL from the Jekyll fixture", () => {
    expect(taxonomyUrls.length).toBeGreaterThan(0);
    for (const url of taxonomyUrls) expect(existsSync(`dist${url}index.html`), url).toBe(true);
  });

  it("builds no category or tag page beyond the fixture", () => {
    for (const kind of ["categories", "tags"]) {
      const built = subdirs(`dist/${kind}`).map((s) => `/${kind}/${s}/`);
      const expected = taxonomyUrls.filter((u) => u.startsWith(`/${kind}/`));
      expect(built.sort()).toEqual(expected.sort());
    }
  });

  it("lists the diffusion posts on /categories/diffusion/", () => {
    const rows = page("/categories/diffusion/").querySelectorAll(".lecture-item");
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) expect(row.getAttribute("href")).toMatch(/^\/posts\/\d{4}\/\d{2}\//);
  });

  it("shows the category tree with counts", () => {
    const tree = page("/categories/");
    expect(tree.querySelector('a[href="/categories/diffusion/"]')).not.toBeNull();
    expect(tree.querySelector(".category-count")?.text).toMatch(/^\d+$/);
  });

  it("shows every tag with a count", () => {
    const links = page("/tags/").querySelectorAll(".tag-list a");
    const tagUrls = taxonomyUrls.filter((u) => u.startsWith("/tags/"));
    expect(links).toHaveLength(tagUrls.length);
    for (const a of links) expect(a.text).toMatch(/\(\d+\)$/);
  });

  it("groups all 20 archive posts by Seoul year, newest year first", () => {
    const doc = page("/archives/");
    const years = doc.querySelectorAll(".archive-year").map((h) => h.text.trim());
    expect(years).toEqual([...years].sort().reverse());
    expect(doc.querySelectorAll(".lecture-item")).toHaveLength(20);
  });

  it("marks the matching navbar link active", () => {
    for (const path of ["/categories/", "/tags/", "/archives/"]) {
      const active = page(path).querySelectorAll(".navbar-link.active");
      expect(active.map((a) => a.getAttribute("href"))).toEqual([path]);
    }
  });
});
