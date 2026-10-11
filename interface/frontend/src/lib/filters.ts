import type { SearchResult } from '../api'

// CJ Notes: Default filter options to be selected
export const DOCUMENT_TYPES = ['Report', 'Guide', 'Article', 'Study', 'Case study']

// CJ Notes: Publication year filter options to be selected, ideally this should be based off of real data from documents
export const SINCE_OPTIONS = [
  { value: 0, label: 'Any time' },
  { value: 2025, label: '2025 or later' },
  { value: 2024, label: '2024 or later' },
  { value: 2023, label: '2023 or later' },
]

export interface Filters {
  types: ReadonlySet<string>
  /** Earliest publication year; 0 means any time. */
  since: number
}

export const DEFAULT_FILTERS: Filters = { types: new Set(DOCUMENT_TYPES), since: 0 }

export const NONE_FILTERS: Filters = {types: new Set(), since: 0}

export const passes = (f: Filters) => (r: SearchResult) => f.types.has(r.type) && r.year >= f.since

export function activeFilterCount(f: Filters): number {
  return (f.types.size < DOCUMENT_TYPES.length ? 1 : 0) + (f.since > 0 ? 1 : 0)
}
