import { useEffect, useId, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  SAMPLE_QUERY,
  getSearchApi,
  type SearchCollection,
  type SearchModel,
  type SearchResult,
} from "./api";
import CollectionsPage from "./collections";
import { Drawer } from "./components/Drawer";
import type { DrawerPage } from "./components/Drawer";
import { FilterButton, FilterPanel } from "./components/Filters";
import { MenuIcon } from "./components/icons";
import { Button } from "./components/buttons";
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
import { useSearchHistory } from "./hooks/useSearchHistory";
import { useSearchPreferences } from "./hooks/useSearchPreferences";
import { useToast } from "./hooks/useToast";
import { useVocabulary } from "./hooks/useVocabulary";
import SearchHistoryPage from "./history";
import SettingsPage from "./settings";
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
const PAGE_SIZE = 10;
const PAGE_BUTTON_COUNT = 5;

type Panel = "vocab" | "filters" | null;

export default function App() {
  const [sampleData, setSampleData] = useSampleDataSetting();
  const { model, setModel, collection, setCollection } = useSearchPreferences();
  const history = useSearchHistory();
  const api = getSearchApi(sampleData);

  const [query, setQuery] = useState(() => (sampleData ? SAMPLE_QUERY : ""));
  const [search, setSearch] = useState<SearchState>(() =>
    sampleData
      ? { status: "loading", query: SAMPLE_QUERY }
      : { status: "idle" },
  );
  const [currentPage, setCurrentPage] = useState(1);
  const [resultTotal, setResultTotal] = useState(0);
  const [activePage, setActivePage] = useState<DrawerPage>("Search");
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

  function runSearch(
    raw: string,
    useSampleData = sampleData,
    useModel: SearchModel = model,
    useCollection: SearchCollection = collection,
    recordHistory = true,
  ) {
    const q = normalize(raw);
    closeFilters();
    if (!q) {
      searchSeq.current++;
      setSearch({ status: "idle" });
      setCurrentPage(1);
      setResultTotal(0);
      inputRef.current?.focus();
      return;
    }
    if (recordHistory) history.add(q);
    setCurrentPage(1);
    setResultTotal(0);
    setSearch({ status: "loading", query: q });
    loadResults(q, useSampleData, 1, useModel, useCollection);
  }

  /** Fetches results and related vocabulary for `q`. Only the newest request gets to update state. */
  async function loadResults(
    q: string,
    useSampleData: boolean,
    page: number,
    useModel: SearchModel,
    useCollection: SearchCollection,
    refreshVocabulary = true,
  ) {
    const seq = ++searchSeq.current;
    const source = getSearchApi(useSampleData);
    try {
      const [response, groups] = await Promise.all([
        source.search(
          q,
          vocab.activeTerms.map((t) => t.label),
          page,
          PAGE_SIZE,
          useModel,
          useCollection,
        ),
        refreshVocabulary
          ? source.vocabulary(q).catch(() => [])
          : Promise.resolve(null),
      ]);
      if (seq !== searchSeq.current) return;
      if (groups !== null) vocab.load(groups);
      setSearch({ status: "done", query: q, results: response.results ?? [] });
      setCurrentPage(page);
      setResultTotal(response.total);
      highlights.sweepIn("all");
    } catch (error) {
      if (seq !== searchSeq.current) return;
      setSearch({
        status: "error",
        query: q,
        message:
          error instanceof Error
            ? error.message
            : "Check the connection and search again.",
      });
    }
  }

  function goToPage(page: number) {
    if (
      search.status !== "done" ||
      page < 1 ||
      page > pageCount ||
      page === currentPage
    )
      return;
    setSearch({ status: "loading", query: search.query });
    loadResults(search.query, sampleData, page, model, collection, false);
  }

  // With sample data on, the page opens with the sample query already searched.
  useEffect(() => {
    if (sampleData) loadResults(SAMPLE_QUERY, true, 1, model, collection);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- first load only
  }, []);

  function changeDataSource(useSampleData: boolean) {
    setSampleData(useSampleData);
    const lastQuery = search.status === "idle" ? "" : search.query;
    if (lastQuery) {
      runSearch(lastQuery, useSampleData, model, collection, false);
    } else if (useSampleData) {
      setQuery(SAMPLE_QUERY);
      runSearch(SAMPLE_QUERY, true, model, collection, false);
    }
  }

  function changeModel(nextModel: SearchModel) {
    setModel(nextModel);
    if (sampleData) setSampleData(false);
    if (search.status !== "idle") {
      runSearch(search.query, false, nextModel, collection, false);
    }
  }

  function changeCollection(nextCollection: SearchCollection) {
    setCollection(nextCollection);
    if (sampleData) setSampleData(false);
    if (search.status !== "idle") {
      runSearch(search.query, false, model, nextCollection, false);
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
  const pageCount = Math.ceil(resultTotal / PAGE_SIZE);
  const firstPage = Math.max(
    1,
    Math.min(
      currentPage - Math.floor(PAGE_BUTTON_COUNT / 2),
      pageCount - PAGE_BUTTON_COUNT + 1,
    ),
  );
  const pageNumbers = Array.from(
    { length: Math.min(PAGE_BUTTON_COUNT, pageCount) },
    (_, index) => firstPage + index,
  );
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
        {activePage === "Search" && (
          <h1 className="sr-only">Document search</h1>
        )}
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

          {activePage === "Search" ? (
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
                onClear={() =>
                  vocab.clear().forEach((key) => highlights.fadeOut(key))
                }
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
          ) : (
            <h1 className="col-span-2 self-center text-[20px] font-semibold text-ink">
              {activePage}
            </h1>
          )}
        </header>

        {activePage === "Search" ? (
          <>
            <main className={ROW}>
              <div className="col-2 min-w-0 max-sm:col-span-full">
                <ResultsHead
                  search={search}
                  shown={shown.length}
                  pageTotal={results.length}
                  total={resultTotal}
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
            {search.status === "done" && pageCount > 1 && (
              <nav
                aria-label="Search results pages"
                className="flex flex-wrap items-center justify-center gap-2 pt-6"
              >
                <Button
                  size="sm"
                  disabled={currentPage === 1}
                  aria-label="First Page"
                  onClick={() => goToPage(1)}
                >
                  Start
                </Button>
                <Button
                  size="sm"
                  disabled={currentPage === 1}
                  aria-label="Previous page"
                  onClick={() => goToPage(currentPage - 1)}
                >
                  Previous
                </Button>
                {pageNumbers.map((page) => (
                  <Button
                    key={page}
                    size="sm"
                    aria-label={`Page ${page}`}
                    aria-current={page === currentPage ? "page" : undefined}
                    className={
                      page === currentPage
                        ? "border-ink bg-hover-paper bg-slate-600"
                        : ""
                    }
                    onClick={() => goToPage(page)}
                  >
                    {page}
                  </Button>
                ))}
                <Button
                  size="sm"
                  disabled={currentPage === pageCount}
                  aria-label="Next page"
                  onClick={() => goToPage(currentPage + 1)}
                >
                  Next
                </Button>
                <Button
                  size="sm"
                  disabled={currentPage === pageCount}
                  aria-label="Last Page"
                  onClick={() => goToPage(pageCount)}
                >
                  End
                </Button>
                <span
                  className="ml-2 text-[13px] text-graphite"
                  aria-live="polite"
                >
                  Page {currentPage} of {pageCount}
                </span>
              </nav>
            )}
          </>
        ) : (
          <main className="mx-auto mt-10 w-full max-w-3xl">
            {activePage === "Collections" && <CollectionsPage />}
            {activePage === "Search history" && (
              <SearchHistoryPage
                entries={history.entries}
                onSearch={(previousQuery) => {
                  setQuery(previousQuery);
                  setActivePage("Search");
                  runSearch(
                    previousQuery,
                    sampleData,
                    model,
                    collection,
                    false,
                  );
                }}
                onRemove={history.remove}
                onClearDate={history.clearDate}
                onClearAll={history.clearAll}
              />
            )}
            {activePage === "Settings" && (
              <SettingsPage
                model={model}
                collection={collection}
                sampleData={sampleData}
                onModelChange={changeModel}
                onCollectionChange={changeCollection}
                onSampleDataChange={changeDataSource}
              />
            )}
          </main>
        )}
      </div>

      <Drawer
        id={drawerId}
        open={drawerOpen}
        currentPage={activePage}
        returnFocusRef={menuButtonRef}
        sampleData={sampleData}
        onSampleDataChange={changeDataSource}
        onNavigate={(page) => {
          setPanel(null);
          setActivePage(page);
        }}
        onClose={() => setDrawerOpen(false)}
      />
      <Toast message={toast.message} visible={toast.visible} />
    </>
  );
}
