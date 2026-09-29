import { useMemo, useRef, useState } from 'react'
import type { HighlightColor, VocabularyGroup } from '../api'
import { normalize } from '../lib/text'

export const KEYWORD_COLOR: HighlightColor = 'lavender'

/** Terms are matched and stored case-insensitively. */
export const termKey = (term: string) => term.toLowerCase()

export interface Keyword {
  key: string
  label: string
}

export interface ActiveTerm {
  key: string
  label: string
  color: HighlightColor
}

export type AddKeywordResult =
  | { kind: 'empty' | 'duplicate' }
  | { kind: 'added'; key: string }
  | { kind: 'suggested'; key: string; label: string; turnedOn: boolean }

interface VocabularyState {
  groups: VocabularyGroup[]
  active: ReadonlySet<string>
  /** User-added terms, in the order they were added. Always active. */
  keywords: Keyword[]
}

function suggestedTerms(groups: VocabularyGroup[]) {
  const terms = new Map<string, { label: string; color: HighlightColor; on: boolean }>()
  for (const group of groups) {
    for (const t of group.terms) terms.set(termKey(t.term), { label: t.term, color: group.color, on: t.on })
  }
  return terms
}

/**
 * Suggested related terms (grouped, each on or off) plus the user's own
 * keywords. Actions read the latest state from a ref, so they stay correct when
 * called after an await.
 */
export function useVocabulary() {
  const [state, setState] = useState<VocabularyState>({ groups: [], active: new Set(), keywords: [] })
  const latest = useRef(state)

  function commit(next: VocabularyState) {
    latest.current = next
    setState(next)
  }

  const suggested = useMemo(() => suggestedTerms(state.groups), [state.groups])

  /** Active terms in display order: suggested groups first, then keywords. */
  const activeTerms = useMemo(() => {
    const terms: ActiveTerm[] = []
    for (const [key, t] of suggested) {
      if (state.active.has(key)) terms.push({ key, label: t.label, color: t.color })
    }
    for (const k of state.keywords) {
      if (state.active.has(k.key)) terms.push({ key: k.key, label: k.label, color: KEYWORD_COLOR })
    }
    return terms
  }, [suggested, state.active, state.keywords])

  /** Anything that isn't a suggested term is a keyword, including ones just removed and still fading out. */
  const colorOf = (key: string): HighlightColor => suggested.get(key)?.color ?? KEYWORD_COLOR

  /**
   * Swaps in the vocabulary for a new query. Terms the user has already seen keep
   * their on/off state; new terms start at their default. Keywords are kept unless
   * the new vocabulary suggests the same term.
   */
  function load(groups: VocabularyGroup[]) {
    const prev = latest.current
    const known = suggestedTerms(prev.groups)
    const next = suggestedTerms(groups)
    const active = new Set<string>()
    for (const [key, t] of next) {
      const seen = known.has(key) || prev.keywords.some((k) => k.key === key)
      if (seen ? prev.active.has(key) : t.on) active.add(key)
    }
    const keywords = prev.keywords.filter((k) => !next.has(k.key))
    for (const k of keywords) active.add(k.key)
    commit({ groups, active, keywords })
  }

  /** Returns whether the term is now on. */
  function toggle(key: string): boolean {
    const prev = latest.current
    const active = new Set(prev.active)
    const on = !active.delete(key)
    if (on) active.add(key)
    commit({ ...prev, active })
    return on
  }

  function addKeyword(raw: string): AddKeywordResult {
    const label = normalize(raw)
    if (!label) return { kind: 'empty' }
    const key = termKey(label)
    const prev = latest.current
    if (prev.keywords.some((k) => k.key === key)) return { kind: 'duplicate' }

    const existing = suggestedTerms(prev.groups).get(key)
    const turnedOn = !prev.active.has(key)
    commit({
      ...prev,
      active: new Set(prev.active).add(key),
      keywords: existing ? prev.keywords : [...prev.keywords, { key, label }],
    })
    return existing ? { kind: 'suggested', key, label: existing.label, turnedOn } : { kind: 'added', key }
  }

  function removeKeyword(key: string) {
    const prev = latest.current
    const active = new Set(prev.active)
    active.delete(key)
    commit({ ...prev, active, keywords: prev.keywords.filter((k) => k.key !== key) })
  }

  /** Back to the suggested defaults, dropping keywords. Returns the keys that turned on. */
  function reset(): Set<string> {
    const prev = latest.current
    const active = new Set<string>()
    for (const [key, t] of suggestedTerms(prev.groups)) if (t.on) active.add(key)
    commit({ ...prev, active, keywords: [] })
    return new Set([...active].filter((key) => !prev.active.has(key)))
  }

  return {
    groups: state.groups,
    active: state.active,
    keywords: state.keywords,
    activeTerms,
    colorOf,
    load,
    toggle,
    addKeyword,
    removeKeyword,
    reset,
  }
}
