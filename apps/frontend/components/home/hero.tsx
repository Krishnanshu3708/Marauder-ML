'use client';

import React from 'react';
import Link from 'next/link';
import { Sparkle, SquaresFour } from '@phosphor-icons/react';
import { BackgroundEffect } from '@/components/effects/background-effect';
import { buttonClass } from '@/components/ui/button';
import { useBlockDissolveNavigate } from '@/lib/effects/use-block-dissolve-navigate';
import { useTranslations } from '@/lib/i18n';

export default function Hero() {
  const { t } = useTranslations();
  const dashboardLink = useBlockDissolveNavigate();

  return (
    <section className="cf-home-hero relative isolate min-h-screen w-full p-4 md:p-12 lg:p-20">
      <BackgroundEffect />
      <div className="cf-home-panel relative flex min-h-[78vh] w-full flex-col items-center justify-center border border-ink px-6 py-16 text-primary bg-canvas shadow-sw-xl md:px-12">
        <div className="cf-home-eyebrow hero-enter mb-6 inline-flex items-center gap-2 border px-4 py-2 font-mono text-xs font-bold uppercase tracking-[0.2em]">
          <span className="cf-status-dot" aria-hidden="true" />
          Resume intelligence, reimagined
        </div>
        <h1 className="cf-home-title hero-enter mb-6 text-center font-mono text-5xl font-bold uppercase leading-none tracking-tighter md:text-7xl lg:text-8xl selection:bg-primary selection:text-white">
          {t('home.brandLine1')}
          <br />
          {t('home.brandLine2')}
        </h1>

        <p className="cf-home-description hero-enter hero-enter-delay-2 mb-8 max-w-2xl text-center text-base leading-relaxed md:text-lg">
          I Solemnly Swear I Am Qualified — Intelligent Resume Matching, TF-IDF Text Similarity &amp; Skill Gap Discovery Engine.
        </p>

        <div className="hero-enter hero-enter-delay-2 flex flex-col gap-4 md:flex-row md:gap-4">
          <Link
            href="/careerfit-ml"
            className={buttonClass({ variant: 'outline', size: 'lg', className: 'gap-2' })}
          >
            <Sparkle aria-hidden="true" className="size-4" />
            Marauder ML Studio
          </Link>
          <Link
            href="/dashboard"
            className={buttonClass({ size: 'lg', className: 'gap-2' })}
            onClick={dashboardLink.onClick}
            onMouseEnter={dashboardLink.prepare}
            onFocus={dashboardLink.prepare}
          >
            <SquaresFour aria-hidden="true" className="size-4" />
            {t('home.launchApp')}
          </Link>
        </div>

        <div className="hero-enter hero-enter-delay-2 mt-8 flex items-center justify-center gap-3 font-mono text-xs uppercase tracking-wide text-steel">
          <span className="font-bold text-ink">Marauder ML</span>
          <span>•</span>
          <span>Machine Learning Mini-Project</span>
          <span>•</span>
          <span>Resume Matching &amp; Skill Gap Analysis</span>
        </div>
      </div>
    </section>
  );
}
