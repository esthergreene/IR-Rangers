# Frontend

The document search interface, built with React, TypeScript, Vite and Tailwind CSS v4. It's a port of the original single-file search prototype.

## Running it

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # type-check and build to dist/
npm run lint
```

## Sample data vs. the backend

The app can run on synthesized prototype data or use the FastAPI backend. Turn **Sample data** on or off at the bottom of the menu (☰). Each browser remembers the choice.

- **On** (the default): queries return demonstration documents after a short delay. Changing the model or collection in Settings automatically turns sample data off and reruns the current query against the backend.
- **Off**: requests go to the backend client in `src/api/backend.ts`. Start the backend with `bash scripts/run.sh --interface` from the repository root; without a server, the results area shows "Search didn't respond."
- Search history is stored on the current device and capped at the 100 most recent searches.
- Save results into named topic collections from the search page. Collections and their saved document details are stored in the current browser, and can be viewed, removed, or deleted from the Collections page.

Optional environment variables (put them in `.env.local`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `/api` | Base URL for backend requests |
| `VITE_USE_SAMPLE_DATA` | `true` | Starting value of the toggle, until someone changes it in the menu |

## Backend contract

`src/api/types.ts` defines the `SearchApi` interface. The sample data and the backend client both implement it. The backend client expects these endpoints to return JSON:

| Request | Response |
| --- | --- |
| `GET /search?q=<query>&page=<n>&page_size=<n>&model=<name>&collection=<name>&term=<t>…` | `{ query, results: SearchResult[], total, model, collection }` |
| `GET /document/<id>?collection=<paper\|paragraph>` | `{ id, title, type, source, year, content }` |
| `GET /vocabulary?q=<query>` | `VocabularyGroup[]` |
| `GET /suggest?q=<partial query>` | `string[]` |

`term` carries the related terms and keywords that are switched on when the search runs.

## Layout

```
src/
  api/          SearchApi types, sample data, backend client
  hooks/        vocabulary state, highlight animation, collections, preferences, toast
  components/   search box, vocabulary bar/panel, filters, results, collection save dialog, drawer
  collections.tsx   saved topic collections page
  lib/          text matching and filter helpers
  index.css     design tokens (light/dark) exposed to Tailwind via @theme
  App.tsx       page layout and the state that ties the pieces together
```
