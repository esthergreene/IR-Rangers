import type { ReactNode } from "react";
import { useId } from "react";
import { Button } from "./components/buttons";
import type { SearchCollection, SearchModel } from "./api";

interface SettingsPageProps {
  model: SearchModel;
  collection: SearchCollection;
  sampleData: boolean;
  onModelChange(model: SearchModel): void;
  onCollectionChange(collection: SearchCollection): void;
  onSampleDataChange(enabled: boolean): void;
}

const selectedButton = "border-ink bg-row-active";

export default function SettingsPage({
  model,
  collection,
  sampleData,
  onModelChange,
  onCollectionChange,
  onSampleDataChange,
}: SettingsPageProps) {
  return (
    <section aria-labelledby="settings-title">
      <p className="text-[15px] text-graphite">
        Choose how searches are ranked and which documents are searched.
      </p>

      <div className="mt-7 grid gap-6">
        <SettingGroup title="Retrieval model" description="Choose how matching documents are ranked.">
          <ChoiceButton selected={model === "bm25"} onClick={() => onModelChange("bm25")}>
            BM25
          </ChoiceButton>
          <ChoiceButton selected={model === "tfidf"} onClick={() => onModelChange("tfidf")}>
            TF-IDF
          </ChoiceButton>
        </SettingGroup>

        <SettingGroup title="Search collection" description="Search complete papers or individual paragraphs.">
          <ChoiceButton selected={collection === "paper"} onClick={() => onCollectionChange("paper")}>
            Papers
          </ChoiceButton>
          <ChoiceButton selected={collection === "paragraph"} onClick={() => onCollectionChange("paragraph")}>
            Paragraphs
          </ChoiceButton>
        </SettingGroup>

        <SettingGroup
          title="Sample data"
          description="Use built-in demonstration results instead of the search backend."
        >
          <SampleDataSwitch
            checked={sampleData}
            onChange={onSampleDataChange}
          />
        </SettingGroup>
        {sampleData && (
          <p className="-mt-4 text-[13px] text-graphite">
            The model and collection settings take effect when sample data is turned off.
          </p>
        )}
      </div>
    </section>
  )
}

function SettingGroup({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <fieldset className="rounded-row border border-rule bg-paper p-4">
      <legend className="px-1 text-[16px] font-semibold text-ink">{title}</legend>
      <p className="mb-3 text-[13px] text-graphite">{description}</p>
      <div className="flex flex-wrap gap-2">{children}</div>
    </fieldset>
  )
}

function ChoiceButton({
  selected,
  children,
  onClick,
}: {
  selected: boolean
  children: ReactNode
  onClick(): void
}) {
  return (
    <Button
      size="sm"
      aria-pressed={selected}
      className={selected ? selectedButton : ""}
      onClick={onClick}
    >
      {children}
    </Button>
  )
}

function SampleDataSwitch({
  checked,
  onChange,
}: {
  checked: boolean
  onChange(enabled: boolean): void
}) {
  const descriptionId = useId()

  return (
    <label className="flex cursor-pointer items-center gap-3">
      <span id={descriptionId} className="text-[14px] font-medium text-ink-2">
        {checked ? "On" : "Off"}
      </span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        aria-label="Sample data"
        aria-describedby={descriptionId}
        onChange={(event) => onChange(event.target.checked)}
        className="peer sr-only"
      />
      <span
        aria-hidden="true"
        className={`relative h-6 w-10 flex-none rounded-full transition-colors duration-150 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-focus ${
          checked ? "bg-primary" : "bg-rule-strong"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 size-5 rounded-full bg-paper shadow-sm transition-[translate] duration-150 ${
            checked ? "translate-x-4" : ""
          }`}
        />
      </span>
    </label>
  )
}
