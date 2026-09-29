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

The backend isn't built yet, so the app can run on the synthesized data from the prototype. Turn **Sample data** on or off at the bottom of the menu (☰). Each browser remembers the choice.

- **On** (the default): every query returns the same six sample documents after a short delay, so you can try search, suggestions, vocabulary and filters.
- **Off**: requests go to the backend client in `src/api/backend.ts`. With no server running they fail, and the results area shows "Search didn't respond."

Optional environment variables (put them in `.env.local`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `/api` | Base URL for backend requests |
| `VITE_USE_SAMPLE_DATA` | `true` | Starting value of the toggle, until someone changes it in the menu |

## Backend contract

`src/api/types.ts` defines the `SearchApi` interface. The sample data and the backend client both implement it. The backend client expects these endpoints to return JSON:

| Request | Response |
| --- | --- |
| `GET /search?q=<query>&term=<t>&term=<t>…` | `{ query, results: SearchResult[] }` |
| `GET /vocabulary?q=<query>` | `VocabularyGroup[]` |
| `GET /suggest?q=<partial query>` | `string[]` |

`term` carries the related terms and keywords that are switched on when the search runs.

## Layout

```
src/
  api/          SearchApi types, sample data, backend client
  hooks/        vocabulary state, highlight animation, sample-data setting, toast
  components/   search box, vocabulary bar/panel, filters, results, drawer
  lib/          text matching and filter helpers
  index.css     design tokens (light/dark) exposed to Tailwind via @theme
  App.tsx       page layout and the state that ties the pieces together
```
