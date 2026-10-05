"""
Build the paper and paragraph collections used for indexing.

Reads every per-paper JSON file, runs the text through the shared Tokenizer,
and writes one JSON object per line (JSONL) for each collection:

    papers.jsonl       one document per paper (title + all non-empty paragraphs)
    paragraphs.jsonl   one document per non-empty paragraph

Also written to the same folder:

    preprocessing_config.yaml   copy of the config used, so queries can be
                                tokenized the same way at search time
    manifest.json               settings and counts for this build
    empty_paragraph_ids.txt     paragraphs left out because no tokens remained

Usage, from the repository root:
    python scripts/build_collections.py
    python scripts/build_collections.py --limit 200 --output data/processed_sample
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

from src.preprocessing.tokenizer import DEFAULT_CONFIG_PATH, Tokenizer  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "JSON Files"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the paper and paragraph collections.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="Folder of per-paper JSON files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder to write the collections to")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Preprocessing config file to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N files (for quick tests)")
    parser.add_argument("--title-weight", type=int, default=1,
                        help="How many times the title's tokens are counted in a paper document")
    parser.add_argument("--paragraph-title", action="store_true",
                        help="Add the paper title's tokens to each paragraph document")
    return parser.parse_args()


def paper_year(paper_id):
    """Return the year from an arXiv ID in YYMM.NNNNN format, e.g. 2408.00001 -> 2024."""
    return 2000 + int(paper_id[:2])


def write_jsonl_line(f, record):
    """Write one record as a single line of JSON."""
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.title_weight < 0:
        sys.exit("--title-weight must be 0 or more.")

    tokenizer = Tokenizer(config_path=args.config)

    files = sorted(args.data.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No .json files found in {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Write to temporary files and rename them at the end, so an interrupted
    # run never leaves half-written collections that look complete.
    papers_tmp = args.output / "papers.jsonl.tmp"
    paragraphs_tmp = args.output / "paragraphs.jsonl.tmp"

    counts = {"papers": 0, "paragraphs": 0, "empty_papers": 0,
              "paper_tokens": 0, "paragraph_tokens": 0}
    empty_paragraph_ids = []
    unreadable_files = []

    start = time.perf_counter()
    with open(papers_tmp, "w", encoding="utf-8") as papers_out, \
         open(paragraphs_tmp, "w", encoding="utf-8") as paragraphs_out:

        for file_number, path in enumerate(files, start=1):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    paper = json.load(f)
            except json.JSONDecodeError:
                unreadable_files.append(path.name)
                continue

            paper_id = paper["paper_id"]
            title = paper.get("title", "")
            title_tokens = tokenizer.tokenize(title)

            # The abstract field is skipped: paragraph 1 repeats it, 
            # so including both would count the abstract twice.
            paper_tokens = title_tokens * args.title_weight  # a new list, safe to extend

            for paragraph_id, text in paper.get("paragraphs", {}).items():
                tokens = tokenizer.tokenize(text)
                if not tokens:
                    empty_paragraph_ids.append(paragraph_id)
                    continue

                # Paper documents get the paragraph's own tokens, without the
                # title prepended, so the title is not repeated once per paragraph.
                paper_tokens.extend(tokens)

                if args.paragraph_title:
                    tokens = title_tokens + tokens

                write_jsonl_line(paragraphs_out, {
                    "doc_id": paragraph_id,
                    "paper_id": paper_id,
                    "text": text,      # original text, might use for snippets in the interface?
                    "tokens": tokens,
                })
                counts["paragraphs"] += 1
                counts["paragraph_tokens"] += len(tokens)

            if not paper_tokens:
                counts["empty_papers"] += 1
                continue

            write_jsonl_line(papers_out, {
                "doc_id": paper_id,
                "title": title,
                "abstract": paper.get("abstract", ""),   # for display only
                "year": paper_year(paper_id),
                "tokens": paper_tokens,
            })
            counts["papers"] += 1
            counts["paper_tokens"] += len(paper_tokens)

            if file_number % 5000 == 0:
                print(f"  {file_number:,} / {len(files):,} files "
                      f"({time.perf_counter() - start:.0f}s)")

    elapsed = time.perf_counter() - start

    papers_tmp.replace(args.output / "papers.jsonl")
    paragraphs_tmp.replace(args.output / "paragraphs.jsonl")
    shutil.copyfile(args.config, args.output / "preprocessing_config.yaml")

    with open(args.output / "empty_paragraph_ids.txt", "w", encoding="utf-8") as f:
        for paragraph_id in empty_paragraph_ids:
            f.write(paragraph_id + "\n")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(args.data),
        "config": str(args.config),
        "settings": {
            "title_weight": args.title_weight,
            "paragraph_title": args.paragraph_title,
            "limit": args.limit,
        },
        "counts": {
            **counts,
            "empty_paragraphs": len(empty_paragraph_ids),
            "unreadable_files": len(unreadable_files),
        },
        "unreadable_files": unreadable_files,
        "seconds": round(elapsed, 1),
    }
    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {counts['papers']:,} papers and {counts['paragraphs']:,} paragraphs "
          f"to {args.output} in {elapsed:.1f}s")
    print(f"Skipped {len(empty_paragraph_ids):,} empty paragraphs, "
          f"{counts['empty_papers']:,} empty papers, {len(unreadable_files):,} unreadable files")


if __name__ == "__main__":
    main()