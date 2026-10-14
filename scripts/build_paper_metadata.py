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
PARAGRAPH_PATH = PROCESSED_DIR / "paragraphs.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "paper_metadata.jsonl"

def load_paragraph_abstracts():
    abstracts = {}

    with open(PARAGRAPH_PATH, "r", encoding="utf-8") as source:
        for line in source:
            paragraph = json.loads(line)

            if paragraph["doc_id"] == paragraph["paper_id"] + "_1":
                abstracts[paragraph["paper_id"]] = paragraph["text"]

    return abstracts

def main():
    count = 0
    paragraph_abstracts = load_paragraph_abstracts()

    with open(INPUT_PATH, "r", encoding="utf-8") as source, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as destination:

        for line in source:
            paper = json.loads(line)

            metadata = {
                "doc_id": paper["doc_id"],
                "title": paper["title"],
                "abstract": paragraph_abstracts.get(paper["doc_id"]) or paper["abstract"],
                "year": paper["year"],
            }

            destination.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            count += 1

    print(f"Generated metadata for {count} papers.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()