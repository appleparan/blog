import { outsideCode } from './mask';

const YOUTUBE = /\{%\s*youtube\s+"([^"]+)"\s*%\}/g;

/** `{% youtube "url" %}` becomes `<YouTube id="url" />`. */
export function convertYoutube(body: string): string {
  return outsideCode(body, (text) =>
    text.replace(YOUTUBE, (_, url: string) => `<YouTube id="${url}" />`),
  );
}
