import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import {
  adjacentPosts,
  categoryIndex,
  categoryTree,
  postSlug,
  postUrl,
  relatedPosts,
  seoulYearMonth,
  slugify,
  tagIndex,
  trendingTags,
  type Post,
} from "../src/lib/posts";

function mk(id: string, date: string, categories: string[] = [], tags: string[] = []): Post {
  return {
    id,
    data: { title: id, date: new Date(date), categories, tags, draft: false },
  };
}

describe("slugify", () => {
  it("lowercases and joins words with hyphens", () => {
    expect(slugify("Server Engineering")).toBe("server-engineering");
    expect(slugify("LLM Inference")).toBe("llm-inference");
  });

  it("strips trailing punctuation like Jekyll", () => {
    expect(slugify("external forcing$$")).toBe("external-forcing");
  });

  it("maps every published tag and category to a URL in the Jekyll fixture", () => {
    const { urls } = JSON.parse(readFileSync("tests/fixtures/jekyll-reference.json", "utf8"));
    const known = new Set<string>(urls);
    const missing: string[] = [];
    for (const file of readdirSync("_posts")) {
      const fm = parse(readFileSync(`_posts/${file}`, "utf8").split("---")[1]);
      if (fm.draft) continue;
      for (const c of fm.categories ?? []) {
        if (!known.has(`/categories/${slugify(c)}/`)) missing.push(`category ${c}`);
      }
      for (const t of fm.tags ?? []) {
        if (!known.has(`/tags/${slugify(t)}/`)) missing.push(`tag ${t}`);
      }
    }
    expect(missing).toEqual([]);
  });
});

describe("post URLs", () => {
  it("strips the date prefix from the id", () => {
    expect(postSlug("2018-07-14-Tox-Workflow")).toBe("Tox-Workflow");
  });

  it("builds the URL from the front matter date, not the filename", () => {
    const post = {
      id: "2018-07-14-Tox-Workflow",
      data: { date: new Date("2017-07-14T00:00:00+09:00") },
    };
    expect(postUrl(post)).toBe("/posts/2017/07/Tox-Workflow/");
  });

  it("uses the Seoul month, not the UTC month", () => {
    expect(seoulYearMonth(new Date("2024-05-01T01:00:00+09:00"))).toEqual({
      year: "2024",
      month: "05",
    });
  });
});

describe("categories", () => {
  const posts = [
    mk("a", "2024-01-01", ["Deep Learning", "Diffusion", "Vision"]),
    mk("b", "2024-02-01", ["Deep Learning", "Diffusion"]),
    mk("c", "2024-03-01", ["Science"]),
  ];

  it("indexes every level of a category list", () => {
    const idx = categoryIndex(posts);
    expect([...idx.keys()].sort()).toEqual(["deep-learning", "diffusion", "science", "vision"]);
    expect(idx.get("diffusion")?.posts).toHaveLength(2);
    expect(idx.get("deep-learning")?.name).toBe("Deep Learning");
  });

  it("builds a two-level tree with counts", () => {
    const tree = categoryTree(posts);
    const dl = tree.find((t) => t.slug === "deep-learning");
    expect(dl?.count).toBe(2);
    expect(dl?.children).toEqual([{ name: "Diffusion", slug: "diffusion", count: 2 }]);
  });
});

describe("tags", () => {
  const posts = [
    mk("a", "2024-01-01", [], ["x", "y"]),
    mk("b", "2024-02-01", [], ["y", "z"]),
    mk("c", "2024-03-01", [], ["y", "z", "w"]),
  ];

  it("indexes tags by slug", () => {
    expect(tagIndex(posts).get("y")?.posts).toHaveLength(3);
  });

  it("orders trending tags by count then name", () => {
    expect(trendingTags(posts).map((t) => t.name)).toEqual(["y", "z", "w", "x"]);
    expect(trendingTags(posts, 2)).toHaveLength(2);
  });
});

describe("relations", () => {
  const a = mk("a", "2024-01-01", ["C1"], ["t1", "t2"]);
  const b = mk("b", "2024-02-01", ["C1"], ["t1"]);
  const c = mk("c", "2024-03-01", ["C1"], ["t1"]);
  const d = mk("d", "2024-04-01", ["Other"], ["none"]);
  const all = [d, c, b, a];

  it("excludes itself and zero scores, ranks by score then newer date", () => {
    const rel = relatedPosts(a, all).map((p) => p.id);
    expect(rel).toEqual(["c", "b"]);
  });

  it("respects the limit", () => {
    expect(relatedPosts(a, all, 1)).toHaveLength(1);
  });

  it("finds adjacent posts with prev as older", () => {
    expect(adjacentPosts(b, all)).toEqual({ prev: a, next: c });
    expect(adjacentPosts(a, all).prev).toBeNull();
    expect(adjacentPosts(d, all).next).toBeNull();
  });
});
