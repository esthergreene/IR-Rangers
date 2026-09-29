import { useEffect, useId, useRef, type RefObject } from 'react'
import { CloseIcon } from './icons'

const NAV_ITEMS = ['Search', 'Saved searches', 'Collections', 'Search history', 'Vocabulary browser', 'Settings']
const CURRENT = 'Search'

interface DrawerProps {
  id: string
  open: boolean
  /** Receives focus when the drawer closes. */
  returnFocusRef: RefObject<HTMLElement | null>
  sampleData: boolean
  onSampleDataChange(enabled: boolean): void
  onNavigate(label: string): void
  onClose(): void
}

export function Drawer({ id, open, returnFocusRef, sampleData, onSampleDataChange, onNavigate, onClose }: DrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null)
  const wasOpen = useRef(open)

  useEffect(() => {
    if (open) closeRef.current?.focus()
    else if (wasOpen.current) returnFocusRef.current?.focus()
    wasOpen.current = open
  }, [open, returnFocusRef])

  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-scrim transition-opacity duration-[240ms] ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
      />
      <aside
        id={id}
        aria-label="Main menu"
        inert={!open}
        className={`fixed inset-y-0 left-0 z-50 flex w-[min(300px,86vw)] flex-col border-r border-rule bg-paper ${
          open
            ? 'visible translate-x-0 shadow-pop [transition:translate_.24s_cubic-bezier(.3,.7,.2,1),visibility_0s]'
            : // Stay visible while sliding out, then hide.
              'invisible -translate-x-full [transition:translate_.24s_cubic-bezier(.3,.7,.2,1),visibility_0s_linear_.24s]'
        }`}
      >
        <div className="flex items-center justify-between border-b border-rule py-3.5 pr-3 pl-5">
          <span className="text-[17px] font-bold">Document search</span>
          <button
            ref={closeRef}
            type="button"
            aria-label="Close menu"
            className="grid size-10 place-items-center rounded-lg hover:bg-hover-paper"
            onClick={onClose}
          >
            <CloseIcon size={20} />
          </button>
        </div>

        <nav className="flex flex-col gap-0.5 p-2.5">
          {NAV_ITEMS.map((label) => (
            <a
              key={label}
              href="#"
              aria-current={label === CURRENT ? 'page' : undefined}
              className="block rounded-md px-3 py-[11px] text-[15px] text-ink-2 hover:bg-hover-paper hover:text-ink aria-[current=page]:bg-row-active aria-[current=page]:font-semibold aria-[current=page]:text-ink"
              onClick={(e) => {
                e.preventDefault()
                onClose()
                if (label !== CURRENT) onNavigate(label)
              }}
            >
              {label}
            </a>
          ))}
        </nav>

        <div className="mt-auto border-t border-rule px-5 py-4">
          <SwitchRow
            label="Sample data"
            description="Show the built-in synthesized results instead of querying the backend."
            checked={sampleData}
            onChange={onSampleDataChange}
          />
        </div>
      </aside>
    </>
  )
}

function SwitchRow({ label, description, checked, onChange }: { label: string; description: string; checked: boolean; onChange(checked: boolean): void }) {
  const descriptionId = useId()
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <span className="min-w-0 flex-1">
        <span className="block text-[15px] font-semibold">{label}</span>
        <span id={descriptionId} className="mt-0.5 block text-[13px] text-graphite">
          {description}
        </span>
      </span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        aria-describedby={descriptionId}
        onChange={(e) => onChange(e.target.checked)}
        className="peer sr-only"
      />
      <span
        aria-hidden="true"
        className={`relative mt-0.5 h-6 w-10 flex-none rounded-full transition-colors duration-150 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-focus ${
          checked ? 'bg-primary' : 'bg-rule-strong'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 size-5 rounded-full bg-paper shadow-sm transition-[translate] duration-150 ${checked ? 'translate-x-4' : ''}`}
        />
      </span>
    </label>
  )
}
