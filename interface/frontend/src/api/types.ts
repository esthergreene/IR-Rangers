export type HighlightColor = 'yellow' | 'mint' | 'pink' | 'lavender'

export interface SearchResult {
  id: string
  type: string
  /** Joins type and source in the meta line: "Report from the Halden Institute". */
  via: 'from' | 'in'
  source: string
  year: number
  format: string
  title: string
  excerpt: string
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
}

export interface VocabularyTerm {
  term: string
  /** Whether the term is turned on by default. */
  on: boolean
}

export interface VocabularyGroup {
  id: string
  label: string
  color: HighlightColor
  terms: VocabularyTerm[]
}

/**
 * Everything the UI needs from a data source. The sample API and the backend
 * client both implement this, so the UI never knows which one it is talking to.
 */
export interface SearchApi {
  /** `terms` are the related terms and keywords that are switched on. */
  search(query: string, terms: string[]): Promise<SearchResponse>
  vocabulary(query: string): Promise<VocabularyGroup[]>
  suggest(text: string): Promise<string[]>
}
