import { normalize } from '../lib/text'
import type { SearchApi, SearchResponse, VocabularyGroup } from './types'

/*
 * Client for the real search backend, which doesn't exist yet. Until it does,
 * every request fails and the UI shows its "Search didn't respond" state.
 *
 * Expected endpoints (JSON responses, shapes in ./types.ts):
 *   GET {base}/search?q=<query>&term=<term>&term=<term>  → SearchResponse
 *   GET {base}/vocabulary?q=<query>                      → VocabularyGroup[]
 *   GET {base}/suggest?q=<partial query>                 → string[]
 *
 * `base` comes from VITE_API_BASE_URL and defaults to /api.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, '')

async function getJson<T>(path: string, params: URLSearchParams): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}?${params}`, {
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) throw new Error(`GET ${path} responded ${res.status}`)
  return (await res.json()) as T
}

export const backendApi: SearchApi = {
  search(query, terms) {
    const params = new URLSearchParams({ q: query })
    for (const term of terms) params.append('term', term)
    return getJson<SearchResponse>('/search', params)
  },

  vocabulary(query) {
    return getJson<VocabularyGroup[]>('/vocabulary', new URLSearchParams({ q: query }))
  },

  suggest(text) {
    return getJson<string[]>('/suggest', new URLSearchParams({ q: normalize(text) }))
  },
}
