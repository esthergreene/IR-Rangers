import { useEffect, useId, useRef, useState, type RefObject } from "react";
import type { SearchResult } from "../api";
import {
  DEFAULT_FILTERS,
  NONE_FILTERS,
  DOCUMENT_TYPES,
  SINCE_OPTIONS,
  passes,
  type Filters,
} from "../lib/filters";
import { plural } from "../lib/text";
import { Button, TextButton } from "./buttons";
import { ChevronDownIcon, FilterIcon } from "./icons";

interface FilterButtonProps {
  ref: RefObject<HTMLButtonElement | null>;
  panelId: string;
  open: boolean;
  activeCount: number;
  onClick(): void;
}

export function FilterButton({
  ref,
  panelId,
  open,
  activeCount,
  onClick,
}: FilterButtonProps) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={
        activeCount ? `Filter results, ${activeCount} active` : "Filter results"
      }
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-controls={panelId}
      className="relative grid h-full w-[50px] flex-none place-items-center rounded-r-lg text-ink-2 hover:bg-hover-paper hover:text-ink focus-visible:rounded-lg focus-visible:-outline-offset-5 aria-expanded:bg-hover-paper aria-expanded:text-ink max-sm:w-11"
      onClick={onClick}
    >
      <FilterIcon filled={activeCount > 0} />
      {/* CJ Notes: Leave below commented out it creates a werid notification thing */}
      {/* {activeCount > 0 && (
        <span className="absolute top-[7px] right-[7px] h-[17px] min-w-[17px] rounded-[9px] bg-primary px-1 text-center text-[11px] leading-[17px] font-bold text-on-primary shadow-[0_0_0_2px_var(--paper)]">
          {activeCount}
        </span>
      )} */}
    </button>
  );
}

interface FilterPanelProps {
  id: string;
  open: boolean;
  applied: Filters;
  results: SearchResult[];
  anchorRef: RefObject<HTMLElement | null>;
  onApply(filters: Filters): void;
  onClose(): void;
}

export function FilterPanel({
  id,
  open,
  applied,
  results,
  anchorRef,
  onApply,
  onClose,
}: FilterPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on any press outside the panel and its toggle button.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (
        !panelRef.current?.contains(target) &&
        !anchorRef.current?.contains(target)
      )
        onClose();
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open, anchorRef, onClose]);

  return (
    <div
      ref={panelRef}
      id={id}
      role="dialog"
      aria-label="Filter results"
      hidden={!open}
      className="absolute top-[60px] right-0 z-25 w-[min(340px,100%)] animate-pop rounded-field border border-rule bg-paper px-4 pt-4 pb-3 shadow-pop max-sm:left-0 max-sm:w-auto"
    >
      {/* Mounted only while open, so edits start from the applied filters each time. */}
      {open && (
        <FilterForm applied={applied} results={results} onApply={onApply} />
      )}
    </div>
  );
}

function FilterForm({
  applied,
  results,
  onApply,
}: Pick<FilterPanelProps, "applied" | "results" | "onApply">) {
  const [staged, setStaged] = useState(applied);
  const firstCheckbox = useRef<HTMLInputElement>(null);
  const sinceId = useId();

  useEffect(() => {
    firstCheckbox.current?.focus();
  }, []);

  const inRange = results.filter((r) => r.year >= staged.since);
  const matching = results.filter(passes(staged)).length;

  function toggleType(type: string, checked: boolean) {
    const types = new Set(staged.types);
    if (checked) types.add(type);
    else types.delete(type);
    setStaged({ ...staged, types });
  }

  return (
    <>
      <fieldset className="mb-3.5 min-w-0">
        <legend className="mb-1.5 block text-[14px] font-semibold">
          Document type
        </legend>
        {DOCUMENT_TYPES.map((type, i) => (
          <label
            key={type}
            className="-mx-1.5 flex min-h-[34px] cursor-pointer items-center gap-2.5 rounded-md px-1.5 text-[15px] hover:bg-hover-paper"
          >
            <input
              ref={i === 0 ? firstCheckbox : undefined}
              type="checkbox"
              checked={staged.types.has(type)}
              onChange={(e) => toggleType(type, e.target.checked)}
              className="m-0 size-[17px] accent-primary"
            />
            <span>{type}</span>
            <span className="ml-auto text-[13px] text-graphite tabular-nums">
              {inRange.filter((r) => r.type === type).length}
            </span>
          </label>
        ))}
      </fieldset>
      {/* CJ Notes: Maybe add a way to choose a date range e.g., 2024-2025, would be a part of SINCE_OPTIONS, ideally SINCE OPTIONS is updated and becomes based off of real data from documents */}
      <div className="mb-3.5">
        <label
          className="mb-1.5 block text-[14px] font-semibold"
          htmlFor={sinceId}
        >
          Published
        </label>
        <div className="relative">
          <select
            id={sinceId}
            value={staged.since}
            onChange={(e) =>
              setStaged({ ...staged, since: Number(e.target.value) })
            }
            className="min-h-10 w-full appearance-none rounded-lg border border-rule-strong bg-paper pr-9 pl-3 text-[15px]"
          >
            {SINCE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <ChevronDownIcon className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-ink-2" />
        </div>
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-rule pt-3">
        <div>
          <TextButton onClick={() => setStaged(NONE_FILTERS)}>
            Clear filters
          </TextButton>
          <TextButton onClick={() => setStaged(DEFAULT_FILTERS)}>
            Select all filters
          </TextButton>
        </div>
        <Button variant="primary" onClick={() => onApply(staged)}>
          Show {plural(matching, "result")}
        </Button>
      </div>
    </>
  );
}
