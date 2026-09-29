import { useState } from 'react'

const STORAGE_KEY = 'ir-rangers:use-sample-data'

function readInitial(): boolean {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored !== null) return stored === 'true'
  } catch {
    // Storage can be unavailable (private windows, blocked site data).
  }
  return import.meta.env.VITE_USE_SAMPLE_DATA !== 'false'
}

/** Whether the UI reads from the synthesized sample data instead of the backend. Remembered per browser. */
export function useSampleDataSetting() {
  const [enabled, setEnabled] = useState(readInitial)

  function update(next: boolean) {
    setEnabled(next)
    try {
      localStorage.setItem(STORAGE_KEY, String(next))
    } catch {
      // Not remembered; the setting still applies for this session.
    }
  }

  return [enabled, update] as const
}
