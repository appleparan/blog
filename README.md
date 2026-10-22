# Jongsu Liam Kim | Blog

Source of <https://blog.liam.kim>. The site is built with
[Astro](https://astro.build) and MDX, and deployed to GitHub Pages by
`.github/workflows/deploy.yml` on every push to `main`.

## Set up

You need [Bun](https://bun.sh) and Node.js 22.12 or later.

```shell
bun install
bunx husky   # once: installs the commit-msg hook
```

Run `bunx husky` yourself: `bun install` does not run the `prepare`
script (checked with Bun 1.4.2).

## Commands

| Command           | What it does                                           |
| ----------------- | ------------------------------------------------------ |
| `bun run dev`     | Start the dev server at `http://localhost:4321`.       |
| `bun run build`   | Build the site into `dist/`.                           |
| `bun run preview` | Serve the built `dist/` folder.                        |
| `bun run test`    | Run the tests once. Run `bun run build` first.         |
| `bun run lint`    | Lint with oxlint.                                      |
| `bun run format`  | Format with oxfmt. `bun run format:check` only checks. |

## Write a post

Add a file `src/content/posts/YYYY-MM-DD-Title.mdx`. The post URL is
`/posts/YYYY/MM/Title/`: the year and month come from `date` in
Asia/Seoul time, and `Title` is the file name without the date, with
its case kept.

```mdx
---
title: My Post
date: 2026-10-23 10:00:00 +0900
categories: [Deep Learning, LLM]
tags: [transformer, attention]
mermaid: true # optional
draft: false # optional
bibliography: 2026-10-23-My-Post.bib
description: One sentence for link previews.
image: /assets/images/post/2026-10-23-My-Post/cover.png
---
```

| Field          | Meaning                                                               |
| -------------- | --------------------------------------------------------------------- |
| `title`        | Required. The post title.                                             |
| `date`         | Required. Publish date with a time zone offset.                       |
| `categories`   | Category path, top level first. Every level gets a page.              |
| `tags`         | Tag list.                                                             |
| `math`         | Optional, no effect. Math (`$...$`, `$$...$$`) renders in every post. |
| `mermaid`      | Optional, default `false`. `true` renders ` ```mermaid ` code blocks. |
| `draft`        | `true` shows the post in `bun run dev` only. The build leaves it out. |
| `bibliography` | File name of the post's `.bib` file in `src/content/bibliography/`.   |
| `description`  | Summary for meta tags. Without it, the start of the post is used.     |
| `image`        | Preview image for social cards.                                       |

Put images in `public/assets/images/post/<post-file-name>/` and link
them as `/assets/images/post/<post-file-name>/<image>`.

### Citations

Put the BibTeX file in `src/content/bibliography/` and name it in the
`bibliography` field. Cite with `[@key]` (or `[@key1; @key2]`). The
reference list is added at the end of the post.

### Components

Import a component at the top of the post, after the front matter.

```mdx
import Figure from '../../components/Figure.astro';
import Quote from '../../components/Quote.astro';
import { YouTube } from '@astro-community/astro-embed-youtube';

<Figure
  src="/assets/images/post/2026-10-23-My-Post/01-plot.png"
  caption="A plot"
  align="center"
  background="#fff"
/>

<Figure
  src="/assets/images/post/2026-10-23-My-Post/02-diagram.png"
  caption={
    <>
      <a href="https://example.com">Source</a>
    </>
  }
  alt="Diagram"
/>

<Quote cite="bibkey">Quoted text. The footer shows the citation for `bibkey`.</Quote>

<YouTube id="https://www.youtube.com/watch?v=VIDEO_ID" />
```

- `Figure`: `src` is required. `caption` is text, or a JSX fragment
  when it has links. `align` is `center` (default), `left`, or
  `right`. `background` fills transparent images. `alt` defaults to a
  text caption.
- `Quote`: `cite` is a BibTeX key from the post's bibliography.
- `YouTube`: `id` is a video URL or ID.

### Widgets

An interactive widget is an Astro component. Put it in
`src/components/`, import it in the post like the components above,
and use the `--fig-*` color variables from `src/styles/global.css` so
it follows the light and dark themes.

## Git history

Do not squash-merge branches into `main`. A post's last-modified date
comes from `git log --follow` on its file, so the history must keep
each file rename.

## License

See [LICENSE](LICENSE).
