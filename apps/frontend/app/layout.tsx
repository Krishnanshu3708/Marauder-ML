import type { Metadata, Viewport } from 'next';
import {
  Lato,
  Noto_Sans_JP,
  Noto_Sans_KR,
  Noto_Sans_SC,
  Raleway,
  Space_Grotesk,
} from 'next/font/google';
import { THEME_COLOR } from '@/lib/theme-color';
import './(default)/css/globals.css';

const spaceGrotesk = Space_Grotesk({
  variable: '--font-space-grotesk',
  subsets: ['latin'],
  display: 'swap',
});

// Lato for the Modern Deedy and Deedy resume templates. Self-hosted at build time like the other
// faces. Lato is a static family, so each weight/style is its own file; `preload: false`
// keeps those files off every page, since only resumes in those templates use them.
// Deedy draws the first name in Hairline (100).
const lato = Lato({
  variable: '--font-lato',
  weight: ['100', '300', '400', '700'],
  style: ['normal', 'italic'],
  subsets: ['latin', 'latin-ext'],
  display: 'swap',
  preload: false,
});

// Raleway for the Deedy template's contact lines, roles and dates. A variable font,
// so one file covers every weight.
const raleway = Raleway({
  variable: '--font-raleway',
  subsets: ['latin', 'latin-ext'],
  display: 'swap',
  preload: false,
});

// CJK fallbacks for Chinese/Japanese/Korean resume content.
const notoSansSC = Noto_Sans_SC({
  variable: '--font-noto-sans-sc',
  display: 'swap',
  preload: false,
});

const notoSansKR = Noto_Sans_KR({
  variable: '--font-noto-sans-kr',
  display: 'swap',
  preload: false,
});

const notoSansJP = Noto_Sans_JP({
  variable: '--font-noto-sans-jp',
  display: 'swap',
  preload: false,
});

export const metadata: Metadata = {
  title: 'Marauder ML — Resume Matching & Skill Gap Analysis',
  description: 'Marauder ML: I Solemnly Swear I Am Qualified — Intelligent Resume Matching & Skill Gap Analysis Engine',
  applicationName: 'Marauder ML',
  keywords: ['resume', 'matcher', 'job', 'application', 'machine learning', 'skill gap', 'marauder ml'],
};

export const viewport: Viewport = { themeColor: THEME_COLOR };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-US" className="h-full" suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} ${lato.variable} ${raleway.variable} ${notoSansSC.variable} ${notoSansKR.variable} ${notoSansJP.variable} antialiased bg-canvas text-ink-soft min-h-full`}
      >
        {children}
      </body>
    </html>
  );
}
