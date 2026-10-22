import { execFileSync } from 'node:child_process';
import { basename, dirname, resolve } from 'node:path';

/** Commits with this trailer value (`Lastmod: skip`) are mechanical, such as the MDX migration. */
const SKIP = 'skip';

/**
 * Date of the latest commit that edited the file, following renames, so a post keeps the
 * history of its original `_posts/<name>.md`. Commits with a `Lastmod: skip` trailer are
 * ignored. Uses the author date: the old history was rewritten, so its committer dates are
 * all the rewrite date. Returns null outside git or when no commit qualifies.
 */
export function lastModified(filePath: string): Date | null {
  const path = resolve(filePath);
  let log: string;
  try {
    log = execFileSync(
      'git',
      ['log', '--follow', '--format=%aI%x09%(trailers:key=Lastmod,valueonly,separator=%x2C)', '--', basename(path)],
      { cwd: dirname(path), encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    );
  } catch {
    return null;
  }
  for (const line of log.split('\n')) {
    const [date, trailer = ''] = line.split('\t');
    if (date && trailer.trim() !== SKIP) return new Date(date);
  }
  return null;
}
