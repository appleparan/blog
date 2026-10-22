import { atomResponse } from '../lib/feed-response';

// jekyll-feed ids: the post URL with its trailing slash.
export const GET = () => atomResponse({ path: '/feed.xml', idStyle: 'url' });
