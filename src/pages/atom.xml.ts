import { atomResponse } from '../lib/feed-response';

// Chirpy ids: no trailing slash, feed id is the bare site URL. Keep them so readers see no new entries.
export const GET = () => atomResponse({ path: '/atom.xml', idStyle: 'chirpy' });
