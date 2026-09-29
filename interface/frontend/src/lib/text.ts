/** Trims and collapses runs of whitespace. */
export function normalize(s: string): string {
  return s.trim().replace(/\s+/g, ' ')
}

export function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Matches any of `keys` as whole words, longest first. Group 1 is the character
 * before the match (or ""), group 2 is the matched term.
 */
export function buildMatcher(keys: Iterable<string>): RegExp | null {
  const alternatives = [...keys].sort((a, b) => b.length - a.length).map(escapeRegExp)
  if (!alternatives.length) return null
  return new RegExp(`(^|[^\\p{L}\\p{N}])(${alternatives.join('|')})(?=$|[^\\p{L}\\p{N}])`, 'giu')
}

export function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export const plural = (n: number, noun: string) => `${n} ${n === 1 ? noun : `${noun}s`}`
