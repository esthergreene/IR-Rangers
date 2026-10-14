import { useState } from "react"

const STORAGE_KEY = "ir-rangers:search-history"
const MAX_HISTORY_ENTRIES = 100

export interface SearchHistoryEntry {
  id: string
  query: string
  searchedAt: string
}

function isEntry(value: unknown): value is SearchHistoryEntry {
  if (typeof value !== "object" || value === null) return false
  const entry = value as Record<string, unknown>
  return (
    typeof entry.id === "string" &&
    typeof entry.query === "string" &&
    typeof entry.searchedAt === "string" &&
    !Number.isNaN(Date.parse(entry.searchedAt))
  )
}

function readInitial(): SearchHistoryEntry[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (!saved) return []
    const parsed: unknown = JSON.parse(saved)
    return Array.isArray(parsed)
      ? parsed.filter(isEntry).slice(0, MAX_HISTORY_ENTRIES)
      : []
  } catch {
    return []
  }
}

export function useSearchHistory() {
  const [entries, setEntries] = useState(readInitial)

  function save(next: SearchHistoryEntry[]) {
    const limited = next.slice(0, MAX_HISTORY_ENTRIES)
    setEntries(limited)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(limited))
    } catch {
      // History remains available for the current session if storage is unavailable.
    }
  }

  function add(query: string) {
    const entry: SearchHistoryEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      query,
      searchedAt: new Date().toISOString(),
    }
    save([entry, ...entries])
  }

  function remove(id: string) {
    save(entries.filter((entry) => entry.id !== id))
  }

  function clearDate(date: string) {
    save(
      entries.filter(
        (entry) => new Date(entry.searchedAt).toLocaleDateString() !== date,
      ),
    )
  }

  function clearAll() {
    save([])
  }

  return { entries, add, remove, clearDate, clearAll }
}
