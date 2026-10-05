import { useEffect, useId, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { SAMPLE_QUERY, getSearchApi, type SearchResult } from "./api";
import { Drawer } from "./components/Drawer";
import { FilterButton, FilterPanel } from "./components/Filters";
import { MenuIcon } from "./components/icons";
import {
  ResultList,
  ResultsHead,
  type HighlightOptions,
  type SearchState,
} from "./components/Results";
import { SearchBox } from "./components/SearchBox";
import { Toast } from "./components/Toast";
import { VocabBar, VocabPanel } from "./components/Vocabulary";
import { useHighlightSweep } from "./hooks/useHighlightSweep";
import { useSampleDataSetting } from "./hooks/useSampleDataSetting";
import { useToast } from "./hooks/useToast";
import { useVocabulary } from "./hooks/useVocabulary";
import {
  DEFAULT_FILTERS,
  activeFilterCount,
  passes,
  type Filters,
} from "./lib/filters";
import { buildMatcher, normalize } from "./lib/text";

/** Three-column grid shared by the header and results: gutter, content, gutter. */
const ROW =
  "grid grid-cols-[var(--gutter)_minmax(0,1fr)_var(--gutter)] items-start gap-x-(--gap)";

const NO_RESULTS: SearchResult[] = [];

type Panel = "vocab" | "filters" | null;

export default function App() {
  const [sampleData, setSampleData] = useSampleDataSetting();
  const api = getSearchApi(sampleData);

  const [query, setQuery] = useState(() => (sampleData ? SAMPLE_QUERY : ""));
  const [search, setSearch] = useState<SearchState>(() =>
    sampleData
      ? { status: "loading", query: SAMPLE_QUERY }
      : { status: "idle" },
  );
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [panel, setPanel] = useState<Panel>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const vocab = useVocabulary();
  const highlights = useHighlightSweep();
  const toast = useToast();

  const inputRef = useRef<HTMLInputElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const filterButtonRef = useRef<HTMLButtonElement>(null);
  const vocabBarRef = useRef<HTMLButtonElement>(null);
  const vocabPanelRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLOListElement>(null);
  const searchSeq = useRef(0);

  const drawerId = useId();
  const vocabPanelId = useId();
  const filterPanelId = useId();

  /* ---------- Search ---------- */

  function runSearch(raw: string, useSampleData = sampleData) {
    const q = normalize(raw);
    closeFilters();
    if (!q) {
      searchSeq.current++;
      setSearch({ status: "idle" });
      inputRef.current?.focus();
      return;
    }
    setSearch({ status: "loading", query: q });
    loadResults(q, useSampleData);
  }

  /** Fetches results and related vocabulary for `q`. Only the newest request gets to update state. */
  async function loadResults(q: string, useSampleData: boolean) {
    const seq = ++searchSeq.current;
    const source = getSearchApi(useSampleData);
    try {
      const [response, groups] = await Promise.all([
        source.search(
          q,
          vocab.activeTerms.map((t) => t.label),
        ),
        source.vocabulary(q).catch(() => []),
      ]);
      if (seq !== searchSeq.current) return;
      vocab.load(groups);
      setSearch({ status: "done", query: q, results: response.results ?? [] });
      highlights.sweepIn("all");
    } catch {
      if (seq !== searchSeq.current) return;
      setSearch({ status: "error", query: q });
    }
  }

  // With sample data on, the page opens with the sample query already searched.
  useEffect(() => {
    if (sampleData) loadResults(SAMPLE_QUERY, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- first load only
  }, []);

  function changeDataSource(useSampleData: boolean) {
    setSampleData(useSampleData);
    const lastQuery = search.status === "idle" ? "" : search.query;
    if (lastQuery) {
      runSearch(lastQuery, useSampleData);
    } else if (useSampleData) {
      setQuery(SAMPLE_QUERY);
      runSearch(SAMPLE_QUERY, true);
    }
  }

  /* ---------- Vocabulary ---------- */

  function toggleTerm(key: string) {
    if (vocab.toggle(key)) highlights.sweepIn(new Set([key]));
    else highlights.fadeOut(key);
  }

  function addKeyword(label: string) {
    const result = vocab.addKeyword(label);
    if (result.kind === "added") {
      highlights.sweepIn(new Set([result.key]));
    } else if (result.kind === "suggested") {
      if (result.turnedOn) highlights.sweepIn(new Set([result.key]));
      toast.show(
        `“${result.label}” is already a suggested term, so it's been turned on.`,
      );
    }
  }

  function removeKeyword(key: string) {
    vocab.removeKeyword(key);
    highlights.fadeOut(key);
  }

  /* ---------- Panels, filters, drawer ---------- */

  function closeFilters(returnFocus = false) {
    setPanel((p) => (p === "filters" ? null : p));
    if (returnFocus) filterButtonRef.current?.focus();
  }

  function closeVocab(returnFocus = false) {
    setPanel((p) => (p === "vocab" ? null : p));
    if (returnFocus) vocabBarRef.current?.focus();
  }

  function applyFilters(next: Filters) {
    setFilters(next);
    closeFilters(true);
  }

  function clearFilters() {
    flushSync(() => setFilters(DEFAULT_FILTERS));
    const firstLink = resultsRef.current?.querySelector("a");
    (firstLink ?? inputRef.current)?.focus();
  }

  function openDrawer() {
    closeFilters();
    setDrawerOpen(true);
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (drawerOpen) {
        setDrawerOpen(false);
      } else if (panel === "filters") {
        closeFilters(true);
      } else if (panel === "vocab") {
        const focused = document.activeElement;
        if (
          focused === vocabBarRef.current ||
          vocabPanelRef.current?.contains(focused)
        )
          closeVocab(true);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  });

  /* ---------- Render ---------- */

  const results = search.status === "done" ? search.results : NO_RESULTS;
  const shown = useMemo(
    () => results.filter(passes(filters)),
    [results, filters],
  );

  const { active } = vocab;
  const { fading } = highlights;
  const matcher = useMemo(
    () => buildMatcher(new Set([...active, ...fading])),
    [active, fading],
  );
  const highlight: HighlightOptions = {
    matcher,
    isLit: (key) => active.has(key) && !highlights.isFresh(key),
    colorOf: vocab.colorOf,
    staggered: highlights.staggered,
    sweeping: highlights.sweeping,
  };

  return (
    <>
      <div
        inert={drawerOpen}
        className="mx-auto max-w-[988px] px-5 pt-9 pb-[72px] [--gap:16px] [--gutter:48px] max-sm:px-3 max-sm:pt-4 max-sm:pb-14 max-sm:[--gap:8px] max-sm:[--gutter:40px]"
      >
        <h1 className="sr-only">Document search</h1>

        <header className={ROW}>
          <button
            ref={menuButtonRef}
            type="button"
            aria-label="Open menu"
            aria-expanded={drawerOpen}
            aria-controls={drawerId}
            className="grid h-[52px] w-(--gutter) place-items-center rounded-field text-ink hover:bg-hover-desk"
            onClick={openDrawer}
          >
            <MenuIcon />
          </button>

          <SearchBox
            value={query}
            onValueChange={setQuery}
            onEdit={() => setPanel(null)}
            onSubmit={(q) => runSearch(q)}
            suggest={api.suggest}
            inputRef={inputRef}
            fieldEnd={
              <FilterButton
                ref={filterButtonRef}
                panelId={filterPanelId}
                open={panel === "filters"}
                activeCount={activeFilterCount(filters)}
                onClick={() =>
                  setPanel((p) => (p === "filters" ? null : "filters"))
                }
              />
            }
          >
            <VocabBar
              ref={vocabBarRef}
              panelId={vocabPanelId}
              open={panel === "vocab"}
              terms={vocab.activeTerms}
              onClick={() => setPanel((p) => (p === "vocab" ? null : "vocab"))}
            />
            <VocabPanel
              ref={vocabPanelRef}
              id={vocabPanelId}
              open={panel === "vocab"}
              groups={vocab.groups}
              active={vocab.active}
              keywords={vocab.keywords}
              onToggle={toggleTerm}
              onAddKeyword={addKeyword}
              onRemoveKeyword={removeKeyword}
              onReset={() => highlights.sweepIn(vocab.reset())}
              onDone={() => closeVocab(true)}
            />
            <FilterPanel
              id={filterPanelId}
              open={panel === "filters"}
              applied={filters}
              results={results}
              anchorRef={filterButtonRef}
              onApply={applyFilters}
              onClose={() => closeFilters()}
            />
          </SearchBox>
        </header>

        <main className={ROW}>
          <div className="col-2 min-w-0 max-sm:col-span-full">
            <ResultsHead
              search={search}
              shown={shown.length}
              total={results.length}
              sampleData={sampleData}
              onClearFilters={clearFilters}
            />
            <ResultList
              ref={resultsRef}
              search={search}
              shown={shown}
              sampleData={sampleData}
              highlight={highlight}
              onOpenDocument={() =>
                toast.show("Document pages aren't connected yet.")
              }
              onClearFilters={clearFilters}
              onUseSampleData={() => changeDataSource(true)}
            />
          </div>
        </main>
      </div>

      <Drawer
        id={drawerId}
        open={drawerOpen}
        returnFocusRef={menuButtonRef}
        sampleData={sampleData}
        onSampleDataChange={changeDataSource}
        onNavigate={(label) => toast.show(`“${label}” isn't connected yet.`)}
        onClose={() => setDrawerOpen(false)}
      />
      <Toast message={toast.message} visible={toast.visible} />
    </>
  );
}
