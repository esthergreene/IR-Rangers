import { normalize } from '../lib/text'
import type { SearchApi, SearchResult, VocabularyGroup } from './types'

/*
 * Synthesized data carried over from the search prototype. Every query returns
 * the same sample results, so the interface can be exercised without a backend.
 */

export const SAMPLE_QUERY = 'remote work'

const SAMPLE_RESULTS: SearchResult[] = [
  {
    id: 'doc-101', type: 'Report', via: 'from', source: 'the Halden Institute', year: 2025, format: 'PDF',
    title: 'Telework Adoption Across Public Agencies, 2020–2025',
    excerpt: 'Agencies that wrote telecommuting into policy early saw fewer service disruptions and steadier staffing through the transition.',
  },
  {
    id: 'doc-102', type: 'Guide', via: 'from', source: 'Northfield Press', year: 2024, format: 'PDF',
    title: 'Managing Distributed Teams: A Field Guide',
    excerpt: 'Written handoffs matter most when distributed teams span several time zones and rarely overlap during the working day.',
  },
  {
    id: 'doc-103', type: 'Article', via: 'in', source: 'Workweek Review', year: 2025, format: 'WEB',
    title: 'The Hybrid Office, Four Years On',
    excerpt: 'Most employers have settled on hybrid work schedules that anchor two or three shared office days each week.',
  },
  {
    id: 'doc-104', type: 'Guide', via: 'from', source: 'the Posture Lab', year: 2023, format: 'PDF',
    title: 'Setting Up a Healthy Home Workspace',
    excerpt: 'People who work from home full time benefit most from a dedicated home office with an adjustable chair and good daylight.',
  },
  {
    id: 'doc-105', type: 'Study', via: 'in', source: 'Retention Quarterly', year: 2022, format: 'PDF',
    title: 'Flexible Schedules and Staff Retention',
    excerpt: 'Staff offered telework alongside other flexible work arrangements were more likely to stay with their employer past two years.',
  },
  {
    id: 'doc-106', type: 'Case study', via: 'from', source: 'Alder & Finch', year: 2024, format: 'DOC',
    title: 'How Virtual Teams Share Knowledge',
    excerpt: 'The virtual teams we followed leaned on asynchronous collaboration in shared documents more than on video calls, and several met in coworking spaces once a month.',
  },
]

const SAMPLE_VOCABULARY: VocabularyGroup[] = [
  {
    id: 'same', label: 'Same meaning', color: 'yellow', terms: [
      { term: 'telework', on: true },
      { term: 'telecommuting', on: true },
      { term: 'work from home', on: true },
    ],
  },
  {
    id: 'related', label: 'Related', color: 'mint', terms: [
      { term: 'hybrid work', on: true },
      { term: 'distributed teams', on: true },
      { term: 'virtual teams', on: true },
      { term: 'flexible work arrangements', on: false },
    ],
  },
  {
    id: 'specific', label: 'More specific', color: 'pink', terms: [
      { term: 'home office', on: false },
      { term: 'coworking', on: false },
      { term: 'asynchronous collaboration', on: false },
    ],
  },
]

const SAMPLE_SUGGESTIONS = [
  'remote work',
  'remote work policy',
  'remote work productivity',
  'remote work and staff retention',
  'remote onboarding',
  'remote team management',
  'hybrid work schedules',
  'telework guidelines',
  'work from home setup',
  'distributed teams across time zones',
]

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export const sampleApi: SearchApi = {
  async search(query) {
    await wait(450)
    return { query, results: SAMPLE_RESULTS }
  },

  async vocabulary() {
    return SAMPLE_VOCABULARY
  },

  async suggest(text) {
    await wait(60)
    const typed = normalize(text)
    const needle = typed.toLowerCase()
    if (!needle) return []
    const starts = SAMPLE_SUGGESTIONS.filter((s) => s.startsWith(needle))
    const within = SAMPLE_SUGGESTIONS.filter((s) => !s.startsWith(needle) && s.includes(needle))
    const found = [...starts, ...within]
    if (found.length) return found.slice(0, 5)
    return [`${typed} policy`, `${typed} research`, `${typed} case studies`]
  },
}
