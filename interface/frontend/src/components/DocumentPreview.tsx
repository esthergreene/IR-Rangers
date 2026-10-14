import { useEffect, useRef } from "react"
import type { DocumentPreview as Preview } from "../api"
import { CloseIcon } from "./icons"

interface DocumentPreviewProps {
  preview: {
    status: "loading" | "done" | "error"
    document: Preview | null
    error: string | null
  }
  onClose(): void
}

export function DocumentPreview({ preview, onClose }: DocumentPreviewProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (!dialog.open) dialog.showModal()
    return () => {
      if (dialog.open) dialog.close()
    }
  }, [])

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="document-preview-title"
      className="m-auto max-h-[min(80vh,800px)] w-[min(720px,calc(100vw-32px))] overflow-hidden rounded-row border border-rule bg-paper p-0 text-ink shadow-pop backdrop:bg-scrim"
      onCancel={(event) => {
        event.preventDefault()
        onClose()
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="flex items-start justify-between gap-4 border-b border-rule px-6 py-5 max-sm:px-4">
        <div className="min-w-0">
          <p className="mb-1 text-[13px] text-graphite">
            {preview.document
              ? `${preview.document.type} · ${preview.document.source} · ${preview.document.year}`
              : "Document preview"}
          </p>
          <h2
            id="document-preview-title"
            className="text-[20px] leading-snug font-semibold max-sm:text-[18px]"
          >
            {preview.document?.title ?? "Loading document…"}
          </h2>
        </div>
        <button
          type="button"
          aria-label="Close document preview"
          className="grid size-9 flex-none place-items-center rounded-lg text-ink-2 hover:bg-hover-paper"
          onClick={onClose}
          autoFocus
        >
          <CloseIcon size={19} />
        </button>
      </div>

      <div className="max-h-[calc(min(80vh,800px)-100px)] overflow-y-auto px-6 py-5 max-sm:px-4">
        {preview.status === "loading" && (
          <p role="status" className="text-[15px] text-graphite">
            Loading preview…
          </p>
        )}
        {preview.status === "error" && (
          <p role="alert" className="text-[15px] text-ink-2">
            {preview.error}
          </p>
        )}
        {preview.status === "done" && preview.document && (
          <p className="whitespace-pre-wrap font-serif text-[16px] leading-[1.75] text-ink-2">
            {preview.document.content || "No preview text is available for this document."}
          </p>
        )}
      </div>
    </dialog>
  )
}
