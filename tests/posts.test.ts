import { describe, expect, it } from 'vitest';
import {
  adjacentPosts,
  categoryIndex,
  compareByDate,
  categoryTree,
  postSlug,
  postUrl,
  relatedPosts,
  seoulYearMonth,
  slugify,
  tagIndex,
  trendingTags,
  type Post,
} from '../src/lib/posts';

function mk(id: string, date: string, categories: string[] = [], tags: string[] = []): Post {
  return {
    id,
    data: { title: id, date: new Date(date), categories, tags, draft: false },
  };
}

describe('slugify', () => {
  it('lowercases and joins words with hyphens', () => {
    expect(slugify('Server Engineering')).toBe('server-engineering');
    expect(slugify('LLM Inference')).toBe('llm-inference');
  });

  it('strips trailing punctuation', () => {
    expect(slugify('external forcing$$')).toBe('external-forcing');
  });

  it('turns each run of separators into one hyphen', () => {
    expect(slugify('a.b')).toBe('a-b');
    expect(slugify('Q&A')).toBe('q-a');
    expect(slugify('foo/bar')).toBe('foo-bar');
    expect(slugify('a - b')).toBe('a-b');
  });
});

describe('post URLs', () => {
  it('strips the date prefix from the id', () => {
    expect(postSlug('2018-07-14-Tox-Workflow')).toBe('Tox-Workflow');
  });

  it('builds the URL from the front matter date, not the filename', () => {
    const post = {
      id: '2018-07-14-Tox-Workflow',
      data: { date: new Date('2017-07-14T00:00:00+09:00') },
    };
    expect(postUrl(post)).toBe('/posts/2017/07/Tox-Workflow/');
  });

  it('uses the Seoul month, not the UTC month', () => {
    expect(seoulYearMonth(new Date('2024-05-01T01:00:00+09:00'))).toEqual({
      year: '2024',
      month: '05',
    });
  });
});

describe('categories', () => {
  const posts = [
    mk('a', '2024-01-01', ['Deep Learning', 'Diffusion', 'Vision']),
    mk('b', '2024-02-01', ['Deep Learning', 'Diffusion']),
    mk('c', '2024-03-01', ['Science']),
  ];

  it('indexes every level of a category list', () => {
    const idx = categoryIndex(posts);
    expect([...idx.keys()].sort()).toEqual(['deep-learning', 'diffusion', 'science', 'vision']);
    expect(idx.get('diffusion')?.posts).toHaveLength(2);
    expect(idx.get('deep-learning')?.name).toBe('Deep Learning');
  });

  it('builds a two-level tree with counts', () => {
    const tree = categoryTree(posts);
    const dl = tree.find((t) => t.slug === 'deep-learning');
    expect(dl?.count).toBe(2);
    expect(dl?.children).toEqual([{ name: 'Diffusion', slug: 'diffusion', count: 2 }]);
  });
});

describe('tags', () => {
  const posts = [
    mk('a', '2024-01-01', [], ['x', 'y']),
    mk('b', '2024-02-01', [], ['y', 'z']),
    mk('c', '2024-03-01', [], ['y', 'z', 'w']),
  ];

  it('indexes tags by slug', () => {
    expect(tagIndex(posts).get('y')?.posts).toHaveLength(3);
  });

  it('orders trending tags by count then name', () => {
    expect(trendingTags(posts).map((t) => t.name)).toEqual(['y', 'z', 'w', 'x']);
    expect(trendingTags(posts, 2)).toHaveLength(2);
  });
});

describe('relations', () => {
  const a = mk('a', '2024-01-01', ['C1'], ['t1', 't2']);
  const b = mk('b', '2024-02-01', ['C1'], ['t1']);
  const c = mk('c', '2024-03-01', ['C1'], ['t1']);
  const d = mk('d', '2024-04-01', ['Other'], ['none']);
  const all = [d, c, b, a];

  it('excludes itself and zero scores, ranks by score then newer date', () => {
    const rel = relatedPosts(a, all).map((p) => p.id);
    expect(rel).toEqual(['c', 'b']);
  });

  it('respects the limit', () => {
    expect(relatedPosts(a, all, 1)).toHaveLength(1);
  });

  it('finds adjacent posts with prev as older', () => {
    expect(adjacentPosts(b, all)).toEqual({ prev: a, next: c });
    expect(adjacentPosts(a, all).prev).toBeNull();
    expect(adjacentPosts(d, all).next).toBeNull();
  });

  it('orders posts with the same date by id, whatever the input order', () => {
    const x = mk('2017-07-17-A', '2017-07-17T12:00:00+09:00');
    const y = mk('2017-07-17-B', '2017-07-17T12:00:00+09:00');
    const older = mk('2017-07-01-Old', '2017-07-01T12:00:00+09:00');
    const newer = mk('2017-08-01-New', '2017-08-01T12:00:00+09:00');
    for (const input of [
      [older, x, y, newer],
      [newer, y, x, older],
    ]) {
      expect(adjacentPosts(x, input)).toEqual({ prev: older, next: y });
      expect(adjacentPosts(y, input)).toEqual({ prev: x, next: newer });
    }
    expect([y, x].sort(compareByDate)).toEqual([x, y]);
    expect([x, y].sort((a, b) => compareByDate(b, a))).toEqual([y, x]);
  });
});
