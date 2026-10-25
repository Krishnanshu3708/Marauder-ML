'use client';

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { ListNumbers, Sparkle, SquaresFour } from '@phosphor-icons/react';
import { buttonClass } from '@/components/ui/button';
import { PageFrame } from '@/components/ui/page-frame';
import { PageHeader } from '@/components/ui/page-header';
import { APP_VERSION, APP_NAME } from '@/lib/config/version';
import { useTranslations } from '@/lib/i18n';

export const SwissGrid = ({
  banner,
  children,
}: {
  /** Page-level alerts, shown inside the frame above the tiles. */
  banner?: React.ReactNode;
  children: React.ReactNode;
}) => {
  const { t } = useTranslations();

  return (
    <PageFrame height="screen">
      <PageHeader>
        <PageHeader.Title>{t('nav.dashboard')}</PageHeader.Title>
        <PageHeader.Subtitle>{t('dashboard.selectModule')}</PageHeader.Subtitle>
      </PageHeader>

      {/* Content Grid - Scrollable area with NO padding.
          @container makes the card grid respond to the container's actual
          width, not the viewport. The Swiss frame is max-w-86rem so on
          ultra-wide screens the cards no longer over-stretch. */}
      <div className="@container flex-1 overflow-y-auto overflow-x-hidden relative z-10">
        <h2 className="sr-only">{t('dashboard.modulesHeading')}</h2>
        {banner && <div className="space-y-4 p-6">{banner}</div>}
        <div className="p-[1.5px]">
          <div className="grid grid-cols-1 @2xl:grid-cols-2 @3xl:grid-cols-3 @5xl:grid-cols-5 bg-ink gap-[1px] border-b border-ink">
            {children}
          </div>
        </div>
      </div>

      {/* Footer - stays above hovered cards */}
      <div className="p-4 bg-canvas flex flex-wrap justify-between items-center gap-4 font-mono text-xs text-primary border-t border-ink shrink-0 relative z-30">
        <div className="flex items-center gap-2">
          <Image src="/logo.svg" alt="" width={20} height={20} className="w-5 h-5" />
          <span className="uppercase font-bold">{APP_NAME}</span>
          <span className="text-steel">v{APP_VERSION}</span>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <Link
            href="/careerfit-ml"
            className={buttonClass({ variant: 'outline', className: 'min-w-36' })}
          >
            <Sparkle aria-hidden="true" />
            Marauder ML
          </Link>
          <Link
            href="/queue"
            className={buttonClass({ variant: 'outline', className: 'min-w-36' })}
          >
            <ListNumbers aria-hidden="true" />
            {t('nav.queue')}
          </Link>
          <Link
            href="/tracker"
            className={buttonClass({ variant: 'outline', className: 'min-w-36' })}
          >
            <SquaresFour aria-hidden="true" />
            {t('nav.applicationTracker')}
          </Link>
          <Link
            href="/settings"
            className={buttonClass({ variant: 'warning', className: 'min-w-36' })}
          >
            {t('nav.settings')}
          </Link>
        </div>
      </div>

      {/* Community strip - a slim second row, same canvas and z-layer as the footer */}
      <div className="px-4 py-2 bg-canvas border-t border-ink shrink-0 relative z-30">
        <div className="flex flex-wrap items-center justify-between text-xs font-mono text-steel w-full">
          <div className="flex items-center gap-2">
            <span className="font-bold uppercase text-ink">Marauder ML</span>
            <span>•</span>
            <span>I Solemnly Swear I Am Qualified</span>
            <span>•</span>
            <span>Resume Matching &amp; Skill Gap Analysis</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/careerfit-ml" className="text-primary hover:underline font-bold uppercase inline-flex items-center gap-1">
              <Sparkle aria-hidden="true" className="size-3.5" />
              ML Studio
            </Link>
          </div>
        </div>
      </div>
    </PageFrame>
  );
};
