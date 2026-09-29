import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react'
import { normalize } from '../lib/text'
import { CloseIcon, SearchIcon, SuggestionIcon } from './icons'

interface SearchBoxProps {
  value: string
  onValueChange(value: string): void
  /** The user typed in the field. */
  onEdit(): void
  onSubmit(value: string): void
  suggest(text: string): Promise<string[]>
  inputRef: RefObject<HTMLInputElement | null>
  /** Rendered at the right end of the field. */
  fieldEnd: ReactNode
  /** Stacked under the field (vocabulary bar and the popover panels). */
  children: ReactNode
}

/** Query field with autocomplete, plus the search button beside it. */
export function SearchBox({ value, onValueChange, onEdit, onSubmit, suggest, inputRef, fieldEnd, children }: SearchBoxProps) {
  const id = useId()
  const inputId = `${id}-q`
  const listId = `${id}-suggestions`
  const optionId = (i: number) => `${id}-suggestion-${i}`

  const [suggestions, setSuggestions] = useState<string[]>([])
  const [activeIndex, setActiveIndex] = useState(-1)
  const requestSeq = useRef(0)
  const listRef = useRef<HTMLUListElement>(null)
  const open = suggestions.length > 0

  useEffect(() => {
    if (activeIndex >= 0) listRef.current?.children[activeIndex]?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  async function updateSuggestions(text: string) {
    const seq = ++requestSeq.current
    if (!normalize(text)) {
      closeSuggestions()
      return
    }
    let list: string[]
    try {
      list = await suggest(text)
    } catch {
      list = []
    }
    if (seq !== requestSeq.current || document.activeElement !== inputRef.current) return
    setSuggestions(list)
    setActiveIndex(-1)
  }

  function closeSuggestions() {
    requestSeq.current++
    setSuggestions([])
    setActiveIndex(-1)
  }

  function submit(text: string) {
    closeSuggestions()
    onSubmit(text)
  }

  function choose(text: string) {
    onValueChange(text)
    submit(text)
  }

  function moveActive(delta: number) {
    const n = suggestions.length
    setActiveIndex((i) => (i < 0 ? (delta > 0 ? 0 : n - 1) : (i + delta + n) % n))
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.nativeEvent.isComposing) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (open) moveActive(1)
      else if (normalize(value)) updateSuggestions(value)
    } else if (e.key === 'ArrowUp') {
      if (open) {
        e.preventDefault()
        moveActive(-1)
      }
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (open && activeIndex >= 0) choose(suggestions[activeIndex])
      else submit(value)
    } else if (e.key === 'Escape' && open) {
      e.preventDefault()
      e.stopPropagation()
      closeSuggestions()
    }
  }

  const needle = normalize(value).toLowerCase()

  return (
    <div role="search" className="col-[2/4] grid grid-cols-[minmax(0,1fr)_var(--gutter)] items-start gap-x-(--gap)">
      <div className="relative z-[5] min-w-0">
        <div className="flex h-[52px] items-center rounded-field border-[1.5px] border-rule-strong bg-paper pl-4 transition-[border-color,box-shadow] duration-150 focus-within:border-ink focus-within:shadow-[0_0_0_4px_var(--hl-yellow)] max-sm:pl-3">
          <label className="sr-only" htmlFor={inputId}>
            Search the collection
          </label>
          <input
            ref={inputRef}
            id={inputId}
            type="text"
            value={value}
            placeholder="Enter your query…"
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            enterKeyHint="search"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={open}
            aria-controls={listId}
            aria-activedescendant={activeIndex >= 0 ? optionId(activeIndex) : undefined}
            className="h-full min-w-0 flex-auto bg-transparent p-0 text-[17px] outline-none placeholder:text-graphite"
            onChange={(e) => {
              onValueChange(e.target.value)
              onEdit()
              updateSuggestions(e.target.value)
            }}
            onKeyDown={onKeyDown}
            onBlur={closeSuggestions}
          />
          {value && (
            <button
              type="button"
              aria-label="Clear query"
              className="relative grid h-full w-11 flex-none place-items-center text-ink-2 hover:text-ink focus-visible:rounded-lg focus-visible:-outline-offset-5 max-sm:w-10"
              onClick={() => {
                onValueChange('')
                closeSuggestions()
                inputRef.current?.focus()
              }}
            >
              <CloseIcon size={18} />
            </button>
          )}
          <span className="h-6 w-px flex-none bg-rule" aria-hidden="true" />
          {fieldEnd}
        </div>

        {children}

        <ul
          ref={listRef}
          id={listId}
          role="listbox"
          aria-label="Suggestions"
          hidden={!open}
          className="absolute top-[calc(100%+6px)] right-0 left-0 z-20 m-0 max-h-80 animate-pop list-none overflow-y-auto rounded-field border border-rule bg-paper p-1.5 shadow-pop"
          // Keep focus in the field while clicking an option.
          onMouseDown={(e) => e.preventDefault()}
        >
          {suggestions.map((s, i) => {
            const at = needle ? s.toLowerCase().indexOf(needle) : -1
            return (
              <li
                key={`${i}:${s}`}
                id={optionId(i)}
                role="option"
                aria-selected={i === activeIndex}
                className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-[9px] text-[16px] text-ink-2 hover:bg-row-active hover:text-ink aria-selected:bg-row-active aria-selected:text-ink"
                onClick={() => choose(s)}
              >
                <SuggestionIcon className="flex-none text-graphite" />
                {at < 0 ? (
                  <b className="font-semibold text-ink">{s}</b>
                ) : (
                  // The typed part stays plain; the completion is bold.
                  <span>
                    {at > 0 && <b className="font-semibold text-ink">{s.slice(0, at)}</b>}
                    {s.slice(at, at + needle.length)}
                    {at + needle.length < s.length && <b className="font-semibold text-ink">{s.slice(at + needle.length)}</b>}
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      </div>

      <button
        type="button"
        aria-label="Search"
        className="grid h-[52px] w-(--gutter) place-items-center rounded-field bg-primary text-on-primary transition-colors duration-150 hover:bg-primary-hover"
        onClick={() => submit(value)}
      >
        <SearchIcon />
      </button>
    </div>
  )
}
