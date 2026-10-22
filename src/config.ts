export const siteConfig = {
  url: 'https://blog.liam.kim',
  title: 'Jongsu Liam Kim | Blog',
  tagline: 'Just a blog',
  description:
    'A record of my studies and insights on topics I enjoy, such as Deep Learning and High-Performance Computing (HPC).',
  lang: 'ko',
  timezone: 'Asia/Seoul',
  author: 'Jongsu Liam Kim',
  avatar: '/assets/commons/avatar.jpg',
  social: {
    github: 'https://github.com/appleparan',
    x: 'https://twitter.com/sky0bserver',
    linkedin: 'https://www.linkedin.com/in/jongsu-kim-63458347',
    email: 'jongsukim8@gmail.com',
  },
  nav: [
    { label: 'Posts', href: '/' },
    { label: 'Categories', href: '/categories/' },
    { label: 'Tags', href: '/tags/' },
    { label: 'Archives', href: '/archives/' },
    { label: 'About', href: '/about/' },
  ],
  postsPerPage: 10,
  // Slots only: comments and analytics stay off until configured.
  comments: { provider: null as string | null },
  analytics: {} as Record<string, string>,
} as const;
