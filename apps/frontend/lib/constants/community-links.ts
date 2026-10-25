/**
 * Where the community and support links point. Plain links only: nothing here is fetched at
 * runtime (no star counts, no avatars), so the app stays local-first.
 */
export const COMMUNITY_LINKS = {
  follow: 'https://github.com/Krishnanshu3708',
  star: 'https://github.com/Krishnanshu3708/Marauder-ML',
  sponsor: 'https://github.com/Krishnanshu3708',
  contributors: 'https://github.com/Krishnanshu3708/Marauder-ML/graphs/contributors',
  discord: 'https://github.com/Krishnanshu3708/Marauder-ML',
} as const;

export type CommunityLinkId = keyof typeof COMMUNITY_LINKS;
