import type { DocumentCollection, SavedDocument } from "./hooks/useCollectionEntry";

interface CollectionsPageProps {
  collections: DocumentCollection[];
  storageError: string | null;
  onOpenDocument(document: SavedDocument): void;
  onRemoveDocument(collectionId: string, documentId: string): void;
  onDeleteCollection(collectionId: string): void;
}

export default function CollectionsPage({
  collections,
  storageError,
  onOpenDocument,
  onRemoveDocument,
  onDeleteCollection,
}: CollectionsPageProps) {
  return (
    <section aria-labelledby="collections-title">
      <div className="mb-6">
        <h2 id="collections-title" className="text-[22px] font-semibold text-ink">
          Your collections
        </h2>
        <p className="mt-1 text-[15px] text-graphite">
          Keep useful documents together by research topic.
        </p>
      </div>
      {storageError && (
        <p
          role="alert"
          className="mb-4 rounded-lg border border-rule-strong bg-hover-paper px-4 py-3 text-[14px] text-ink-2"
        >
          {storageError}
        </p>
      )}
      {collections.length === 0 ? (
        <div className="rounded-row border border-dashed border-rule-strong bg-paper px-6 py-8">
          <p className="text-[16px] font-semibold text-ink">
            No collections yet
          </p>
          <p className="mt-1 text-[14px] text-graphite">
            Search for documents, then choose “Save to collection” to start a
            topic folder.
          </p>
        </div>
      ) : (
        <div className="grid gap-5">
          {collections.map((collection) => (
            <section
              key={collection.id}
              aria-labelledby={`collection-${collection.id}`}
              className="overflow-hidden rounded-row border border-rule bg-paper"
            >
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-rule px-5 py-4">
                <div className="min-w-0">
                  <h3
                    id={`collection-${collection.id}`}
                    className="break-words text-[18px] font-semibold text-ink"
                  >
                    {collection.name}
                  </h3>
                  <p className="mt-0.5 text-[13px] text-graphite">
                    {collection.documents.length}{" "}
                    {collection.documents.length === 1 ? "document" : "documents"}
                  </p>
                </div>
                <button
                  type="button"
                  className="text-[14px] font-medium text-ink-2 underline decoration-rule-strong underline-offset-3 hover:text-ink hover:decoration-current"
                  onClick={() => onDeleteCollection(collection.id)}
                >
                  Delete collection
                </button>
              </header>
              {collection.documents.length === 0 ? (
                <p className="px-5 py-5 text-[14px] text-graphite">
                  This collection is empty. Save documents from your search
                  results to add them here.
                </p>
              ) : (
                <ul className="divide-y divide-rule">
                  {collection.documents.map((document) => (
                    <li
                      key={document.id}
                      className="flex flex-wrap items-center justify-between gap-x-5 gap-y-3 px-5 py-4"
                    >
                      <div className="min-w-0 flex-1">
                        <button
                          type="button"
                          className="text-left text-[16px] font-semibold text-ink hover:underline focus-visible:outline-2 focus-visible:outline-focus"
                          onClick={() => onOpenDocument(document)}
                        >
                          {document.result.title}
                        </button>
                        <p className="mt-1 text-[13px] text-graphite">
                          {document.result.type} {document.result.via}{" "}
                          {document.result.source}, {document.result.year}
                        </p>
                        <p className="mt-2 line-clamp-3 font-serif text-[14px] leading-relaxed text-ink-2">
                          {document.result.excerpt}
                        </p>
                      </div>
                      <button
                        type="button"
                        className="text-[13px] font-medium text-ink-2 underline decoration-rule-strong underline-offset-2 hover:text-ink hover:decoration-current"
                        onClick={() =>
                          onRemoveDocument(collection.id, document.id)
                        }
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          ))}
        </div>
      )}
    </section>
  );
}
