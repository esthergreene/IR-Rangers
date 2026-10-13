"""
This is for generating paper metadata for the FastAPI search backend.

Reads papers.jsonl and saves only the fields needed
to display search results in the frontend.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "papers.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "paper_metadata.jsonl"

def main():
    count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as source, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as destination:

        for line in source:
            paper = json.loads(line)

            metadata = {
                "doc_id": paper["doc_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "year": paper["year"],
            }

            destination.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            count += 1

    print(f"Generated metadata for {count} papers.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()