import { useState } from "react"
import type { SearchCollection, SearchModel } from "../api"

const MODEL_KEY = "ir-rangers:search-model"
const COLLECTION_KEY = "ir-rangers:search-collection"

function readModel(): SearchModel {
  try {
    const stored = localStorage.getItem(MODEL_KEY)
    if (stored === "tfidf") return "tfidf"
  } catch {
    // Keep the default when browser storage is unavailable.
  }
  return "bm25"
}

function readCollection(): SearchCollection {
  try {
    const stored = localStorage.getItem(COLLECTION_KEY)
    if (stored === "paragraph") return "paragraph"
  } catch {
    // Keep the default when browser storage is unavailable.
  }
  return "paper"
}

export function useSearchPreferences() {
  const [model, setModelState] = useState<SearchModel>(readModel)
  const [collection, setCollectionState] = useState<SearchCollection>(readCollection)

  function setModel(next: SearchModel) {
    setModelState(next)
    try {
      localStorage.setItem(MODEL_KEY, next)
    } catch {
      // The preference still applies for this session.
    }
  }

  function setCollection(next: SearchCollection) {
    setCollectionState(next)
    try {
      localStorage.setItem(COLLECTION_KEY, next)
    } catch {
      // The preference still applies for this session.
    }
  }

  return { model, setModel, collection, setCollection }
}
