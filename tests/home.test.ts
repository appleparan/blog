import { existsSync, readFileSync } from "node:fs";
import { parse } from "node-html-parser";
import { describe, expect, it } from "vitest";

// Reads the built site, so run `bun run build` before `bun run test`.
// 20 published posts (2 drafts are dev-only), 10 per page: `/` and `/page/2/`, no `/page/3/`.
const page = (path: string) => parse(readFileSync(`dist${path}index.html`, "utf8"));
const cardUrls = (path: string) =>
  page(path)
    .querySelectorAll(".post-card .post-card-title a")
    .map((a) => a.getAttribute("href"));

describe("home page", () => {
  it("lists 10 cards on / and 10 on /page/2/, and has no /page/3/", () => {
    expect(cardUrls("/")).toHaveLength(10);
    expect(cardUrls("/page/2/")).toHaveLength(10);
    expect(existsSync("dist/page/3/index.html")).toBe(false);
  });

  it("puts the newest published post first and never repeats a post across pages", () => {
    expect(cardUrls("/")[0]).toBe("/posts/2025/11/VAE-to-DDPMs/");
    const all = [...cardUrls("/"), ...cardUrls("/page/2/")];
    expect(new Set(all).size).toBe(20);
  });

  it("shows no draft posts", () => {
    const all = [...cardUrls("/"), ...cardUrls("/page/2/")].join(" ");
    expect(all).not.toContain("Inside-vLLM");
    expect(all).not.toContain("EBM-to-NCSN");
  });

  it("shows meta and an excerpt on every card", () => {
    for (const card of page("/").querySelectorAll(".post-card")) {
      expect(card.querySelector(".post-card-meta")?.text).toMatch(/\d{4}-\d{2}-\d{2}.*\d+분/);
      expect(card.querySelector(".post-card-excerpt")?.text.length).toBeGreaterThan(0);
    }
  });

  it("lists 5 recently updated posts and 10 trending tags in the panel", () => {
    const panel = page("/").querySelector(".side-panel");
    expect(panel?.querySelectorAll(".side-recent a")).toHaveLength(5);
    const tags = panel?.querySelectorAll(".side-tags a") ?? [];
    expect(tags).toHaveLength(10);
    for (const a of tags) expect(a.getAttribute("href")).toMatch(/^\/tags\/[^/]+\/$/);
  });

  it("links pagination between the pages", () => {
    expect(page("/").querySelector('.pagination a[rel="next"]')?.getAttribute("href")).toBe(
      "/page/2/",
    );
    expect(page("/page/2/").querySelector('.pagination a[rel="prev"]')?.getAttribute("href")).toBe(
      "/",
    );
    expect(page("/page/2/").querySelector('.pagination a[rel="next"]')).toBeNull();
  });
});
