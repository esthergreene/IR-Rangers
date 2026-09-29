import type { CSSProperties, ReactNode, Ref } from 'react'
import type { HighlightColor, SearchResult } from '../api'
import { Button, TextButton } from './buttons'
import { DocIcon } from './icons'

export type SearchState =
  | { status: 'idle' }
  | { status: 'loading'; query: string }
  | { status: 'done'; query: string; results: SearchResult[] }
  | { status: 'error'; query: string }

export interface HighlightOptions {
  /** Matches every term that currently has a highlight span (see `buildMatcher`). */
  matcher: RegExp | null
  isLit(key: string): boolean
  colorOf(key: string): HighlightColor
  /** Delay each stroke by its position in the excerpt, for the first sweep over new results. */
  staggered: boolean
  sweeping: boolean
}

const cssVars = (vars: Record<`--${string}`, string>) => vars as CSSProperties

interface ResultsHeadProps {
  search: SearchState
  shown: number
  total: number
  sampleData: boolean
  onClearFilters(): void
}

export function ResultsHead({ search, shown, total, sampleData, onClearFilters }: ResultsHeadProps) {
  let content: ReactNode = null
  if (search.status === 'loading') {
    content = <p className="text-[15px] text-graphite">Searching…</p>
  } else if (search.status === 'done') {
    const filtered = shown !== total
    const noun = total === 1 ? 'result' : 'results'
    content = (
      <>
        <p className="text-[15px] text-graphite">
          <strong className="font-semibold text-ink">{filtered ? `${shown} of ${total} ${noun}` : `${total} ${noun}`}</strong>
          {` for “${search.query}”`}
          {filtered && (
            <TextButton className="ml-1.5" onClick={onClearFilters}>
              Clear filters
            </TextButton>
          )}
        </p>
        {sampleData && (
          <span
            title="Every query returns these sample results. Turn off sample data in the menu to use the backend."
            className="rounded-full border border-rule-strong px-2.5 py-0.5 text-[12.5px] whitespace-nowrap text-graphite"
          >
            Sample data
          </span>
        )}
      </>
    )
  }

  return (
    <div aria-live="polite" className="mt-7 mb-3 flex min-h-[30px] flex-wrap items-center justify-between gap-x-4 gap-y-2 max-sm:mt-5">
      {content}
    </div>
  )
}

interface ResultListProps {
  ref: Ref<HTMLOListElement>
  search: SearchState
  shown: SearchResult[]
  sampleData: boolean
  highlight: HighlightOptions
  onOpenDocument(result: SearchResult): void
  onClearFilters(): void
  onUseSampleData(): void
}

export function ResultList({ ref, search, shown, sampleData, highlight, onOpenDocument, onClearFilters, onUseSampleData }: ResultListProps) {
  let content: ReactNode
  if (search.status === 'loading') {
    content = [0, 1, 2].map((i) => <SkeletonRow key={i} />)
  } else if (search.status === 'idle') {
    content = <EmptyState title="Enter a query to search the collection." body="Results appear here, with related terms highlighted." />
  } else if (search.status === 'error') {
    content = (
      <EmptyState title="Search didn't respond." body="Check the connection and search again.">
        {!sampleData && <Button onClick={onUseSampleData}>Use sample data instead</Button>}
      </EmptyState>
    )
  } else if (!shown.length) {
    content = (
      <EmptyState title="No results match these filters." body="Turn on more document types or pick a wider date range.">
        <Button onClick={onClearFilters}>Clear filters</Button>
      </EmptyState>
    )
  } else {
    content = shown.map((result, i) => (
      <ResultRow key={result.id} result={result} index={i} highlight={highlight} onOpen={onOpenDocument} />
    ))
  }

  return (
    <ol
      ref={ref}
      aria-busy={search.status === 'loading' || undefined}
      className={`grid list-none gap-3 ${highlight.sweeping ? 'sweeping' : ''}`}
    >
      {content}
    </ol>
  )
}

const ROW = 'grid grid-cols-[44px_minmax(0,1fr)] gap-x-5 rounded-row border border-rule bg-paper py-5 pr-6 pl-5 max-sm:grid-cols-[30px_minmax(0,1fr)] max-sm:gap-x-3.5 max-sm:p-4'

function ResultRow({ result, index, highlight, onOpen }: { result: SearchResult; index: number; highlight: HighlightOptions; onOpen(result: SearchResult): void }) {
  return (
    <li className={ROW} style={cssVars({ '--d': `${index * 90}ms` })}>
      <DocIcon format={result.format} />
      <div>
        <h2 className="text-[17px] leading-[1.3] font-semibold max-sm:text-[16px]">
          <a
            href="#"
            className="hover:underline hover:decoration-1 hover:underline-offset-3"
            onClick={(e) => {
              e.preventDefault()
              onOpen(result)
            }}
          >
            {result.title}
          </a>
        </h2>
        <p className="mt-1.5 max-w-[68ch] font-serif text-[16px] leading-[1.62] text-ink-2 max-sm:text-[15.5px]">
          <HighlightedText text={result.excerpt} {...highlight} />
        </p>
        <p className="mt-2.5 text-[13px] text-graphite">
          {result.type} {result.via} {result.source}, {result.year}
        </p>
      </div>
    </li>
  )
}

function HighlightedText({ text, matcher, isLit, colorOf, staggered }: HighlightOptions & { text: string }) {
  if (!matcher) return text

  const parts: ReactNode[] = []
  let last = 0
  let n = 0
  for (const m of text.matchAll(matcher)) {
    const start = m.index + m[1].length
    const word = m[2]
    const key = word.toLowerCase()
    if (start > last) parts.push(text.slice(last, start))
    parts.push(
      <span
        key={`${start}:${key}`}
        data-color={colorOf(key)}
        className={isLit(key) ? 'hl on' : 'hl'}
        style={staggered ? cssVars({ '--hd': `${n * 110}ms` }) : undefined}
      >
        {word}
      </span>,
    )
    last = start + word.length
    n++
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

function SkeletonRow() {
  const bar = 'block animate-skeleton bg-rule'
  return (
    <li className={ROW} aria-hidden="true">
      <span className={`${bar} h-11 w-9 rounded-[4px_10px_4px_4px] max-sm:h-[37px] max-sm:w-[30px]`} />
      <div>
        <span className={`${bar} mt-1 mb-4 h-3.5 w-[52%] rounded`} />
        <span className={`${bar} mb-2.5 h-[11px] w-[92%] rounded`} />
        <span className={`${bar} mb-2.5 h-[11px] w-[64%] rounded`} />
      </div>
    </li>
  )
}

function EmptyState({ title, body, children }: { title: string; body: string; children?: ReactNode }) {
  return (
    <li className="rounded-row border border-dashed border-rule-strong bg-paper px-7 py-9">
      <p className="mb-1 text-[17px] font-semibold text-ink">{title}</p>
      <p className="text-graphite">{body}</p>
      {children && <div className="mt-4">{children}</div>}
    </li>
  )
}
