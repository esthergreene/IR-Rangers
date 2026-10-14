import { useEffect, useRef, useState } from "react";

interface CollectionOption {
  id: string;
  name: string;
}

interface CollectionSaveDialogProps {
  collections: CollectionOption[];
  documentTitle: string;
  onSave(collectionId: string, newCollectionName: string): void;
  onClose(): void;
}

export function CollectionSaveDialog({
  collections,
  documentTitle,
  onSave,
  onClose,
}: CollectionSaveDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [selectedId, setSelectedId] = useState(collections[0]?.id ?? "");
  const [newCollectionName, setNewCollectionName] = useState("");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    return () => {
      if (dialog.open) dialog.close();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="save-document-title"
      className="m-auto w-[min(480px,calc(100vw-32px))] rounded-row border border-rule bg-paper p-0 text-ink shadow-pop backdrop:bg-scrim"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (selectedId) onSave(selectedId, "");
        }}
      >
        <div className="border-b border-rule px-6 py-5 max-sm:px-4">
          <h2 id="save-document-title" className="text-[19px] font-semibold">
            Save document
          </h2>
          <p className="mt-1 text-[14px] text-graphite">{documentTitle}</p>
        </div>
        <div className="grid gap-5 px-6 py-5 max-sm:px-4">
          {collections.length > 0 && (
            <label className="grid gap-1.5 text-[14px] font-medium">
              Save to an existing collection
              <select
                value={selectedId}
                onChange={(event) => setSelectedId(event.target.value)}
                className="min-h-10 rounded-lg border border-rule-strong bg-paper px-3 text-ink"
              >
                {collections.map((collection) => (
                  <option key={collection.id} value={collection.id}>
                    {collection.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          {collections.length > 0 && (
            <div className="flex justify-end">
              <button
                type="submit"
                className="rounded-lg border border-rule-strong px-4 py-2 text-[14px] font-semibold hover:border-ink-2"
              >
                Save to collection
              </button>
            </div>
          )}
          <label className="grid gap-1.5 text-[14px] font-medium">
            Create a topic collection
            <input
              value={newCollectionName}
              onChange={(event) => setNewCollectionName(event.target.value)}
              maxLength={80}
              placeholder="e.g. Remote work policy"
              className="min-h-10 rounded-lg border border-rule-strong bg-paper px-3 text-ink placeholder:text-graphite"
            />
          </label>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              className="rounded-lg px-4 py-2 text-[14px] font-medium text-ink-2 hover:bg-hover-paper"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!newCollectionName.trim()}
              className="rounded-lg border border-primary bg-primary px-4 py-2 text-[14px] font-semibold text-on-primary hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => onSave("", newCollectionName)}
            >
              Create and save
            </button>
          </div>
        </div>
      </form>
    </dialog>
  );
}
