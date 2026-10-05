import {
  useId,
  useRef,
  useState,
  type ReactNode,
  type Ref,
  type RefObject,
} from "react";
import type { HighlightColor, VocabularyGroup } from "../api";
import {
  KEYWORD_COLOR,
  termKey,
  type ActiveTerm,
  type Keyword,
} from "../hooks/useVocabulary";
import { Button, TextButton } from "./buttons";
import { CheckIcon, ChevronDownIcon, CloseIcon, PlusIcon } from "./icons";

interface VocabBarProps {
  ref: RefObject<HTMLButtonElement | null>;
  panelId: string;
  open: boolean;
  terms: ActiveTerm[];
  onClick(): void;
}

/** Summary strip under the search field: which related terms are on. Toggles the panel. */
export function VocabBar({
  ref,
  panelId,
  open,
  terms,
  onClick,
}: VocabBarProps) {
  const count = terms.length;
  return (
    <button
      ref={ref}
      type="button"
      aria-expanded={open}
      aria-controls={panelId}
      aria-label={
        count
          ? `More vocab options. Also searching for ${count} related ${count === 1 ? "term" : "terms"}.`
          : "More vocab options. No related terms are on."
      }
      className="mt-2 flex min-h-[42px] w-full items-center gap-3 rounded-field border border-rule bg-paper py-[5px] pr-2.5 pl-3.5 text-left text-[14px] hover:bg-hover-paper aria-expanded:rounded-b-none max-sm:pl-2.5"
      onClick={onClick}
    >
      {/* CJ Notes: Add ability to remove vocab words by clicking on them */}
      <span
        className={`flex-none text-graphite ${count ? "max-sm:hidden" : ""}`}
      >
        {count ? "Also searching for" : "No related terms are on"}
      </span>
      {/* CJ Notes: Add ability to be able to scroll through selected terms */}
      <span className="flex min-w-0 flex-auto gap-1.5 overflow-hidden whitespace-nowrap [mask-image:linear-gradient(90deg,#000_calc(100%_-_28px),transparent)]">
        {terms.map((t) => (
          <span
            key={t.key}
            data-color={t.color}
            className="flex-none rounded-stroke bg-(--c) px-[7px] py-0.5 text-ink"
          >
            {t.label}
          </span>
        ))}
      </span>
      <span className="inline-flex flex-none items-center gap-1.5 font-semibold text-ink">
        <span className="max-sm:hidden">Vocab Options</span>
        <ChevronDownIcon
          size={18}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </span>
    </button>
  );
}

interface VocabPanelProps {
  ref: Ref<HTMLDivElement>;
  id: string;
  open: boolean;
  groups: VocabularyGroup[];
  active: ReadonlySet<string>;
  keywords: Keyword[];
  onToggle(key: string): void;
  onAddKeyword(label: string): void;
  onRemoveKeyword(key: string): void;
  onReset(): void;
  onDone(): void;
}

export function VocabPanel({
  ref,
  id,
  open,
  groups,
  active,
  keywords,
  onToggle,
  onAddKeyword,
  onRemoveKeyword,
  onReset,
  onDone,
}: VocabPanelProps) {
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();

  function addKeyword() {
    if (!draft.trim()) {
      inputRef.current?.focus();
      return;
    }
    onAddKeyword(draft);
    setDraft("");
  }

  return (
    <div
      ref={ref}
      id={id}
      hidden={!open}
      className="animate-pop rounded-b-field border border-t-0 border-rule bg-paper px-4 pt-1 pb-3.5"
    >
      {groups.map((group) => (
        <VocabGroupRow key={group.id} label={group.label} color={group.color}>
          {(labelId) => (
            <div
              className="flex flex-wrap gap-2"
              role="group"
              aria-labelledby={labelId}
            >
              {group.terms.map((t) => {
                const key = termKey(t.term);
                const on = active.has(key);
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={on}
                    className="inline-flex min-h-8 items-center gap-1.5 rounded-stroke border border-rule-strong py-1 pr-3 pl-[9px] text-left text-[14px] leading-[1.2] text-ink-2 transition-[background-color,border-color] duration-150 hover:border-ink-2 hover:text-ink aria-pressed:border-transparent aria-pressed:bg-(--c) aria-pressed:text-ink aria-pressed:hover:border-ink-2"
                    onClick={() => onToggle(key)}
                  >
                    {on ? <CheckIcon size={14} /> : <PlusIcon size={14} />}
                    <span>{t.term}</span>
                  </button>
                );
              })}
            </div>
          )}
        </VocabGroupRow>
      ))}

      <VocabGroupRow label="Your keywords" color={KEYWORD_COLOR}>
        {(labelId) => (
          <>
            {keywords.length > 0 && (
              <div
                className="mb-2 flex flex-wrap gap-2"
                role="group"
                aria-labelledby={labelId}
              >
                {keywords.map((k) => (
                  <button
                    key={k.key}
                    type="button"
                    aria-label={`Remove keyword ${k.label}`}
                    className="inline-flex min-h-8 items-center gap-1.5 rounded-stroke border border-transparent bg-(--c) py-1 pr-2 pl-3 text-left text-[14px] leading-[1.2] text-ink transition-[border-color] duration-150 hover:border-ink-2"
                    onClick={() => {
                      onRemoveKeyword(k.key);
                      inputRef.current?.focus();
                    }}
                  >
                    <span>{k.label}</span>
                    <CloseIcon size={14} />
                  </button>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-1.5">
              <label className="sr-only" htmlFor={inputId}>
                Add a keyword
              </label>
              <input
                ref={inputRef}
                id={inputId}
                type="text"
                value={draft}
                placeholder="Add a keyword"
                maxLength={40}
                autoComplete="off"
                enterKeyHint="done"
                className="min-h-8 w-[min(220px,100%)] rounded-stroke border border-rule-strong bg-transparent px-2.5 py-1 text-[14px] placeholder:text-graphite focus:border-ink focus:shadow-[0_0_0_3px_var(--hl-lavender)] focus:outline-none max-sm:text-[16px]"
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    addKeyword();
                  }
                }}
              />
              <Button size="sm" onClick={addKeyword}>
                Add
              </Button>
            </div>
          </>
        )}
      </VocabGroupRow>

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5 pt-3">
        <p className="max-w-[44ch] text-[13px] text-graphite">
          Terms that are on get added to your search and highlighted in the
          results.
        </p>
        {/* CJ Notes: Add a button to clear all selected terms */}
        <div className="ml-auto flex items-center gap-3">
          <TextButton onClick={onReset}>Reset to suggested</TextButton>
          <Button onClick={onDone}>Done</Button>
        </div>
      </div>
    </div>
  );
}

function VocabGroupRow({
  label,
  color,
  children,
}: {
  label: string;
  color: HighlightColor;
  children(labelId: string): ReactNode;
}) {
  const labelId = useId();
  return (
    <div
      data-color={color}
      className="grid grid-cols-[136px_minmax(0,1fr)] gap-x-4 gap-y-2 border-b border-rule py-3 max-sm:grid-cols-[minmax(0,1fr)]"
    >
      <div
        id={labelId}
        className="flex min-h-8 items-center gap-[9px] text-[14px] font-semibold max-sm:min-h-0"
      >
        <span
          className="h-2.5 w-[18px] flex-none rounded-[5px_2px_6px_3px] bg-(--c)"
          aria-hidden="true"
        />
        {label}
      </div>
      <div>{children(labelId)}</div>
    </div>
  );
}
