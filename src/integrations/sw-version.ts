import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { AstroIntegration } from "astro";

const TOKEN = "__BUILD_ID__";

/** Git short SHA when available, else a timestamp, so every deploy gets a new cache name. */
function buildId(): string {
  try {
    const sha = execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    if (/^[0-9a-f]+$/.test(sha)) return sha;
  } catch {
    // Not a git checkout or git is missing: use the timestamp.
  }
  return Date.now().toString(36);
}

/** Replaces the `__BUILD_ID__` token in `dist/sw.min.js` so the worker cache name changes per build. */
export function swVersion(): AstroIntegration {
  return {
    name: "sw-version",
    hooks: {
      "astro:build:done": ({ dir }) => {
        const file = fileURLToPath(new URL("sw.min.js", dir));
        const source = readFileSync(file, "utf8");
        if (!source.includes(TOKEN)) throw new Error(`${TOKEN} not found in sw.min.js`);
        writeFileSync(file, source.replaceAll(TOKEN, buildId()));
      },
    },
  };
}
