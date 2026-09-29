import { useEffect, useRef, useState } from 'react'
import { prefersReducedMotion } from '../lib/text'

type Fresh = 'all' | ReadonlySet<string>

const NONE: ReadonlySet<string> = new Set()

/**
 * Animation state for the highlighter strokes in result excerpts.
 *
 * - `fresh` terms render un-highlighted for two frames, then highlight, so the
 *   stroke sweeps in. `'all'` does that for every term (a new set of results).
 * - `fading` terms are no longer active but stay rendered briefly so their
 *   stroke can sweep out before the span is removed.
 * - While `sweeping`, strokes are delayed per row (and per term when
 *   `staggered`) so they cascade down the page.
 */
export function useHighlightSweep() {
  const [fresh, setFresh] = useState<Fresh>(NONE)
  const [fading, setFading] = useState<ReadonlySet<string>>(NONE)
  const [sweep, setSweep] = useState<'all' | 'terms' | null>(null)
  const removeTimer = useRef<number>(undefined)
  const sweepTimer = useRef<number>(undefined)

  useEffect(() => {
    if (fresh !== 'all' && fresh.size === 0) return
    // Two frames: the first lets the un-highlighted spans paint, the second flips them on.
    let inner = 0
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setFresh(NONE))
    })
    return () => {
      cancelAnimationFrame(outer)
      cancelAnimationFrame(inner)
    }
  }, [fresh])

  /** Sweep in the given term keys, or every highlight with `'all'`. Also drops anything still fading. */
  function sweepIn(keys: Fresh) {
    clearTimeout(removeTimer.current)
    setFading(NONE)
    if (keys !== 'all' && keys.size === 0) return

    const motion = !prefersReducedMotion()
    setFresh(keys)
    setSweep(motion ? (keys === 'all' ? 'all' : 'terms') : null)
    clearTimeout(sweepTimer.current)
    if (motion) sweepTimer.current = window.setTimeout(() => setSweep(null), 1500)
  }

  function fadeOut(key: string) {
    setFading((prev) => new Set(prev).add(key))
    clearTimeout(removeTimer.current)
    removeTimer.current = window.setTimeout(() => setFading(NONE), prefersReducedMotion() ? 0 : 340)
  }

  return {
    fading,
    sweeping: sweep !== null,
    staggered: sweep === 'all',
    isFresh: (key: string) => fresh === 'all' || fresh.has(key),
    sweepIn,
    fadeOut,
  }
}
