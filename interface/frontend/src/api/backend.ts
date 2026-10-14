import { normalize } from '../lib/text'
import type { DocumentPreview, SearchApi, SearchResponse, VocabularyGroup } from './types'

/*
 * Client for the real search backend.
 *
 * Expected endpoints (JSON responses, shapes in ./types.ts):
 *   GET {base}/search?q=<query>&page=<n>&page_size=<n>&model=<name>&collection=<name> → SearchResponse
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
  search(query, terms, page, pageSize, model, collection) {
    const params = new URLSearchParams({
      q: query,
      page: String(page),
      page_size: String(pageSize),
      model,
      collection,
    })
    for (const term of terms) params.append('term', term)
    return getJson<SearchResponse>('/search', params).then((response) => {
      if (response.model !== model || response.collection !== collection) {
        throw new Error('The search backend did not apply the selected model and collection. Restart the backend and try again.')
      }
      return response
    })
  },

  preview(id, collection) {
    return getJson<DocumentPreview>(
      `/document/${encodeURIComponent(id)}`,
      new URLSearchParams({ collection }),
    )
  },

  vocabulary(query) {
    return getJson<VocabularyGroup[]>('/vocabulary', new URLSearchParams({ q: query }))
  },

  suggest(text) {
    return getJson<string[]>('/suggest', new URLSearchParams({ q: normalize(text) }))
  },
}
