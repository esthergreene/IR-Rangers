import { useState } from "react";
import type { SearchCollection, SearchResult } from "../api";

const STORAGE_KEY = "ir-rangers:collections";

export interface SavedDocument {
  id: string;
  result: SearchResult;
  searchCollection: SearchCollection;
  sampleData: boolean;
  savedAt: string;
}

export interface DocumentCollection {
  id: string;
  name: string;
  documents: SavedDocument[];
}

function isSearchResult(value: unknown): value is SearchResult {
  if (typeof value !== "object" || value === null) return false;
  const result = value as Record<string, unknown>;
  return (
    typeof result.id === "string" &&
    typeof result.title === "string" &&
    typeof result.type === "string" &&
    (result.via === "from" || result.via === "in") &&
    typeof result.source === "string" &&
    typeof result.year === "number" &&
    typeof result.format === "string" &&
    typeof result.excerpt === "string"
  );
}

function isSavedDocument(value: unknown): value is SavedDocument {
  if (typeof value !== "object" || value === null) return false;
  const saved = value as Record<string, unknown>;
  return (
    typeof saved.id === "string" &&
    isSearchResult(saved.result) &&
    (saved.searchCollection === "paper" ||
      saved.searchCollection === "paragraph") &&
    typeof saved.sampleData === "boolean" &&
    typeof saved.savedAt === "string"
  );
}

function isDocumentCollection(value: unknown): value is DocumentCollection {
  if (typeof value !== "object" || value === null) return false;
  const collection = value as Record<string, unknown>;
  return (
    typeof collection.id === "string" &&
    typeof collection.name === "string" &&
    collection.name.trim().length > 0 &&
    Array.isArray(collection.documents) &&
    collection.documents.every(isSavedDocument)
  );
}

function readInitial(): { collections: DocumentCollection[]; error: string | null } {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return { collections: [], error: null };
    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed) || !parsed.every(isDocumentCollection)) {
      return {
        collections: [],
        error: "Saved collections are in an unrecognized format.",
      };
    }
    return { collections: parsed, error: null };
  } catch {
    return {
      collections: [],
      error: "Saved collections could not be read from this browser.",
    };
  }
}

function createId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function useCollectionEntry() {
  const [initial] = useState(readInitial);
  const [collections, setCollections] = useState(initial.collections);
  const [error, setError] = useState(initial.error);

  function save(next: DocumentCollection[]) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      const message = "Collections could not be saved to this browser.";
      setError(message);
      throw new Error(message);
    }
    setCollections(next);
    setError(null);
  }

  function saveDocument(
    result: SearchResult,
    searchCollection: SearchCollection,
    sampleData: boolean,
    collectionId: string,
    newCollectionName: string,
  ) {
    let target = collections.find((item) => item.id === collectionId);
    let next = collections;

    if (!target) {
      const name = newCollectionName.trim();
      if (!name) throw new Error("Enter a name for the new collection.");
      target = collections.find(
        (item) => item.name.toLocaleLowerCase() === name.toLocaleLowerCase(),
      );
      if (!target) {
        target = { id: createId(), name, documents: [] };
        next = [...collections, target];
      }
    }

    const alreadySaved = target.documents.some(
      (item) =>
        item.result.id === result.id &&
        item.searchCollection === searchCollection &&
        item.sampleData === sampleData,
    );
    if (alreadySaved) return { collectionName: target.name, alreadySaved: true };

    const document: SavedDocument = {
      id: createId(),
      result,
      searchCollection,
      sampleData,
      savedAt: new Date().toISOString(),
    };
    save(
      next.map((item) =>
        item.id === target?.id
          ? { ...item, documents: [...item.documents, document] }
          : item,
      ),
    );
    return { collectionName: target.name, alreadySaved: false };
  }

  function removeDocument(collectionId: string, documentId: string) {
    save(
      collections.map((item) =>
        item.id === collectionId
          ? {
              ...item,
              documents: item.documents.filter(
                (document) => document.id !== documentId,
              ),
            }
          : item,
      ),
    );
  }

  function deleteCollection(collectionId: string) {
    save(collections.filter((item) => item.id !== collectionId));
  }

  return {
    collections,
    error,
    saveDocument,
    removeDocument,
    deleteCollection,
  };
}
