import { existsSync, readFileSync } from "node:fs";
import { parse } from "node-html-parser";
import { describe, expect, it } from "vitest";
import { buildAtom } from "../src/lib/feed";
import type { Post } from "../src/lib/posts";

const post = (over: Partial<Post["data"]> = {}): Post => ({
  id: "2024-03-05-Tom-and-Jerry",
  body: "First paragraph of the body.",
  data: {
    title: "Tom & <Jerry>",
    date: new Date("2024-03-05T00:00:00Z"),
    categories: [],
    tags: ["a b", "c"],
    draft: false,
    ...over,
  },
});

describe("buildAtom", () => {
  const xml = buildAtom([post()], "https://blog.liam.kim");

  it("escapes & and < in titles", () => {
    expect(xml).toContain("Tom &amp; &lt;Jerry&gt;");
    expect(xml).not.toContain("Tom & <Jerry>");
  });

  it("uses absolute URLs for id and link", () => {
    const url = "https://blog.liam.kim/posts/2024/03/Tom-and-Jerry/";
    expect(xml).toContain(`<id>${url}</id>`);
    expect(xml).toContain(`<link href="${url}"`);
    expect(xml).toContain('<link rel="self" type="application/atom+xml" href="https://blog.liam.kim/feed.xml"');
  });

  it("has feed metadata, a summary, and a category per tag", () => {
    expect(xml).toContain("<title>Jongsu Liam Kim | Blog</title>");
    expect(xml).toContain("<updated>2024-03-05T00:00:00.000Z</updated>");
    expect(xml).toContain("<summary>First paragraph of the body.</summary>");
    expect(xml).toContain('<category term="a b"/>');
    expect(xml).toContain('<category term="c"/>');
  });
});

describe("built site", () => {
  it("serves the same Atom feed at /feed.xml and /atom.xml", () => {
    const feed = readFileSync("dist/feed.xml", "utf8");
    expect(feed.startsWith("<?xml")).toBe(true);
    expect(readFileSync("dist/atom.xml", "utf8")).toBe(feed);
  });

  it("writes the sitemap index and a copy at /sitemap.xml", () => {
    expect(existsSync("dist/sitemap-index.xml")).toBe(true);
    expect(readFileSync("dist/sitemap.xml", "utf8")).toBe(
      readFileSync("dist/sitemap-index.xml", "utf8"),
    );
  });

  it("leaves the 404 page out of the sitemap", () => {
    const sitemap = readFileSync("dist/sitemap-0.xml", "utf8");
    expect(sitemap).toContain("https://blog.liam.kim/about/");
    expect(sitemap).not.toMatch(/\/404(\.html|\/)/);
  });

  it("adds canonical, Open Graph, and the feed link to every page", () => {
    const root = parse(readFileSync("dist/about/index.html", "utf8"));
    expect(root.querySelector("link[rel=canonical]")?.getAttribute("href")).toBe(
      "https://blog.liam.kim/about/",
    );
    expect(root.querySelector("meta[property='og:type']")?.getAttribute("content")).toBe("website");
    expect(root.querySelector("link[type='application/atom+xml']")?.getAttribute("href")).toBe(
      "/feed.xml",
    );
    expect(root.querySelector("script[type='application/ld+json']")).toBeNull();
  });

  it("adds BlogPosting JSON-LD and og:type article on posts", () => {
    const root = parse(readFileSync("dist/posts/2017/07/Tox-Workflow/index.html", "utf8"));
    expect(root.querySelector("meta[property='og:type']")?.getAttribute("content")).toBe("article");
    const ld = JSON.parse(root.querySelector("script[type='application/ld+json']")!.text);
    expect(ld["@type"]).toBe("BlogPosting");
    expect(ld.url).toBe("https://blog.liam.kim/posts/2017/07/Tox-Workflow/");
    expect(ld.datePublished).toMatch(/^2017-07/);
    expect(ld.dateModified).toBeTruthy();
    expect(ld.author.name).toBe("Jongsu Liam Kim");
    expect(root.querySelector("meta[property='og:image']")?.getAttribute("content")).toMatch(
      /^https:\/\/blog\.liam\.kim\//,
    );
  });
});
