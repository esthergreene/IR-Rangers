"""
Build the cleaned-text paragraph collection used by the neural models (Part 2).

The Part 1 collections (build_collections.py) store stemmed tokens for BM25 and 
TF-IDF. Neural models need natural text, so this script only cleans the text 
(math, URLs, citation markers, whitespace) and keeps the words as written.

Writes to the output folder:
    paragraphs.jsonl            one document per non-empty paragraph
    preprocessing_config.yaml   copy of the config used, so queries can be cleaned 
                                the same way at search time
    manifest.json               settings, counts, and length statistics
    empty_paragraph_ids.txt     paragraphs left out because no text remained

There is no papaer collection: Paper rankings are built from paragraph scores 
(MaxP), so each paragraph keeps its paper_id.

Usage, from the repository root:
    python scripts/build_text_collections.py
    python scripts/build_text_collections.py --limit 200 --output 
data/processed_neural_sample
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocessing_neural.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "JSON Files"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed_neural"

def parse_args():
    parser = argparse.ArgumentParser(description="Build the paragraph collection.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="Folder of per-paper JSON files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder to write the collections to")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Preprocessing config file to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N files (for quick tests)")

    return parser.parse_args()

def write_jsonl_line(f, record):
    """Write one record as a single line of JSON."""
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

def percentile(sorted_values, p):
    """Return the p-th percentile (0-100) of an already sorted list, or 0 if it is empty"""
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * p / 
                                                         100))]

def main():
    args = parse_args()
    cleaning = load_cleaning_config(args.config)

    files = sorted(args.data.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No .json files found in {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Write to a temp file and rename it at the end, so an interrupted run never leaves 
    # a half-written collection that looks complete.
    paragraphs_tmp = args.output / "paragraphs.jsonl.tml"

    counts = {"papers": 0, "paragraphs": 0, "empty_papers": 0}
    word_lengths = []
    empty_paragraph_ids = []
    unreadable_files = []

    start = time.perf_counter()
    with open(paragraphs_tmp, "w", encoding="utf-8") as paragraphs_out:
        for file_number, path in enumerate(files, start=1):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    paper = json.load(f)
            except json.JSONDecodeError:
                unreadable_files.append(path.name)
                continue

            paper_id = paper["paper_id"]
            title = clean_for_encoder(paper.get("title", ""), **cleaning)
            paragraphs_written = 0

            # The abstract field is skipped, as in part 1: paragraph 1 repeats it
            for paragraph_id, raw_text in paper.get("paragraphs", {}).items():
                text = clean_for_encoder(raw_text, **cleaning)
                if not text:
                    empty_paragraph_ids.append(paragraph_id)
                    continue

                # The title is stored next to the text rather than prepended, so 
                # each model can decide whether to use it without a rebuild
                write_jsonl_line(paragraphs_out, {
                    "doc_id": paragraph_id, 
                    "paper_id": paper_id,
                    "title": title,
                    "text": text,
                })
                paragraphs_written += 1
                word_lengths.append(len(text.split()))

            counts["papers"] += 1
            counts["paragraphs"] += paragraphs_written
            if paragraphs_written == 0:
                counts["empty_papers"] += 1

            if file_number % 5000 == 0:
                print(f"  {file_number:,} / {len(files)} files "
                      f"({time.perf_counter() - start:.0f}s)")
    elapsed = time.perf_counter() - start

    paragraphs_tmp.replace(args.output / "paragraphs.jsonl")
    shutil.copyfile(args.config, args.output / "preprocessing_config.yaml")

    with open(args.output / "empty_paragraph_ids.txt", "w", encoding="utf-8") as f:
        for paragraph_id in empty_paragraph_ids:
            f.write(paragraph_id + "\n")

    # Lengths are in whitespace-separated words. Subword tokenizers produce more tokens 
    # than this (more still for math heavy text), so treat these as a lower bound when 
    # choosing a model's maximum sequence length
    word_lengths.sort()
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(args.data),
        "config": str(args.config),
        "cleaning": cleaning,
        "settings": {"limit": args.limit},
        "counts": {
            **counts,
            "empty_paragraphs": len(empty_paragraph_ids),
            "unreadable_files": len(unreadable_files),
        },
        "paragraph_words": {
            "median": percentile(word_lengths, 50), 
            "p90": percentile(word_lengths, 90),
            "p99": percentile(word_lengths, 99),
            "max": word_lengths[-1] if word_lengths else 0,
        },
        "unreadable_files": unreadable_files,
        "seconds": round(elapsed, 1),
    }
    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {counts['paragraphs']:,} paragraphs from {counts['papers']:,} papers"
          f"to {args.output} in {elapsed:.1f}s")
    print(f"Skipped {len(empty_paragraph_ids):,} empty paragraphs, "
          f"{counts['empty_papers']:,} papers with no paragraphs, "
          f"{len(unreadable_files):,} unreadable files")

if __name__ == "__main__":
    main()