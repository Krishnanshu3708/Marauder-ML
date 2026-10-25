import createDOMPurify from 'dompurify';

/**
 * Whitelist of allowed HTML tags for rich text content
 */
const ALLOWED_TAGS = ['strong', 'em', 'u', 'a'];

/**
 * Whitelist of allowed HTML attributes
 */
const ALLOWED_ATTR = ['href', 'target', 'rel'];

/**
 * Fallback sanitizer for server-side rendering (SSR) in Node.js environments
 * where a browser DOM or compatible jsdom is unavailable (e.g. Node 20 LTS).
 * Strictly preserves only whitelisted tags: <strong>, <em>, <u>, and safe <a> tags.
 */
function sanitizeServerHtml(dirty: string): string {
  if (!dirty) return '';

  // 1. Strip script and style blocks completely
  let clean = dirty
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');

  // 2. Filter tags against whitelist
  clean = clean.replace(/<\/?([a-z0-9]+)\b([^>]*)>/gi, (match, tag, attrs) => {
    const lowerTag = tag.toLowerCase();
    if (!ALLOWED_TAGS.includes(lowerTag)) {
      return '';
    }

    if (match.startsWith('</')) {
      return `</${lowerTag}>`;
    }

    if (lowerTag !== 'a') {
      return `<${lowerTag}>`;
    }

    // For <a>: strictly extract href, target, rel
    const hrefMatch = attrs.match(/\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
    const targetMatch = attrs.match(/\btarget\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
    const relMatch = attrs.match(/\brel\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);

    let safeAttrs = '';
    if (hrefMatch) {
      const rawHref = (hrefMatch[1] || hrefMatch[2] || hrefMatch[3] || '').trim();
      if (!/^(?:javascript|data|vbscript):/i.test(rawHref)) {
        safeAttrs += ` href="${encodeURI(rawHref)}"`;
      }
    }
    if (targetMatch) {
      const rawTarget = (targetMatch[1] || targetMatch[2] || targetMatch[3] || '').trim();
      if (['_blank', '_self', '_parent', '_top'].includes(rawTarget)) {
        safeAttrs += ` target="${rawTarget}"`;
      }
    }
    if (relMatch) {
      const rawRel = (relMatch[1] || relMatch[2] || relMatch[3] || '').trim();
      safeAttrs += ` rel="${rawRel}"`;
    }

    return `<a${safeAttrs}>`;
  });

  return clean;
}

/**
 * Sanitizes HTML content using DOMPurify with a strict whitelist.
 * Only allows bold, italic, underline, and link formatting.
 * Uses native DOMPurify in the browser and safe whitelist SSR in Node.js.
 *
 * @param dirty - The unsanitized HTML string
 * @returns Sanitized HTML string safe for rendering
 */
export function sanitizeHtml(dirty: string): string {
  if (!dirty) return '';

  if (typeof window !== 'undefined') {
    const domPurify = createDOMPurify(window);
    return domPurify.sanitize(dirty, {
      ALLOWED_TAGS,
      ALLOWED_ATTR,
      FORCE_BODY: true,
    });
  }

  return sanitizeServerHtml(dirty);
}
