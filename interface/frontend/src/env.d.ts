interface ImportMetaEnv {
  /** Base URL of the search backend. Defaults to /api. */
  readonly VITE_API_BASE_URL?: string
  /** Set to "false" to start with sample data off. The in-app toggle overrides this. */
  readonly VITE_USE_SAMPLE_DATA?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
