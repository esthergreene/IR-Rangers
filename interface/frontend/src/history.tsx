import { useMemo } from "react"
import { Button, TextButton } from "./components/buttons"
import type { SearchHistoryEntry } from "./hooks/useSearchHistory"

interface SearchHistoryPageProps {
  entries: SearchHistoryEntry[]
  onSearch(query: string): void
  onRemove(id: string): void
  onClearDate(date: string): void
  onClearAll(): void
}

export default function SearchHistoryPage({
  entries,
  onSearch,
  onRemove,
  onClearDate,
  onClearAll,
}: SearchHistoryPageProps) {
  const groups = useMemo(() => {
    const byDate = new Map<string, SearchHistoryEntry[]>()
    for (const entry of entries) {
      const date = new Date(entry.searchedAt).toLocaleDateString()
      const group = byDate.get(date) ?? []
      group.push(entry)
      byDate.set(date, group)
    }
    return [...byDate.entries()]
  }, [entries])

  return (
    <section aria-labelledby="history-title">
      <div className="items-end justify-between gap-3">
        <Button size="sm" disabled={!entries.length} onClick={onClearAll}>
          Clear all history
        </Button>
      </div>

      {!entries.length ? (
        <div className="mt-6 rounded-row border border-dashed border-rule-strong bg-paper px-6 py-8">
          <p className="font-semibold text-ink">No searches yet</p>
          <p className="mt-1 text-[14px] text-graphite">
            Searches you submit will be saved here on this device.
          </p>
        </div>
      ) : (
        <div className="mt-6 grid gap-4">
          {groups.map(([date, searches]) => (
            <section
              key={date}
              aria-label={`Searches from ${date}`}
              className="rounded-row border border-rule bg-paper px-5 py-4"
            >
              <div className="flex items-center justify-between gap-3 border-b border-rule pb-3">
                <h3 className="text-[17px] font-semibold text-ink">
                  {formatDate(date)}
                </h3>
                <TextButton onClick={() => onClearDate(date)}>
                  Clear day
                </TextButton>
              </div>
              <ul className="m-0 list-none divide-y divide-rule p-0">
                {searches.map((entry) => (
                  <li
                    key={entry.id}
                    className="flex items-center justify-between gap-4 py-2.5"
                  >
                    <button
                      type="button"
                      className="min-w-0 flex-1 truncate text-left text-[15px] text-ink hover:underline"
                      onClick={() => onSearch(entry.query)}
                      title={`Search again for: ${entry.query}`}
                    >
                      {entry.query}
                    </button>
                    <span className="shrink-0 text-[12px] text-graphite">
                      {new Date(entry.searchedAt).toLocaleTimeString([], {
                        hour: "numeric",
                        minute: "2-digit",
                      })}
                    </span>
                    <TextButton
                      aria-label={`Remove search: ${entry.query}`}
                      onClick={() => onRemove(entry.id)}
                    >
                      Remove
                    </TextButton>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </section>
  )
}

function formatDate(date: string): string {
  const today = new Date().toLocaleDateString()
  const yesterday = new Date(Date.now() - 86_400_000).toLocaleDateString()
  if (date === today) return "Today"
  if (date === yesterday) return "Yesterday"
  return date
}
