"""
This file handles the backend API for our search engine.

The frontend sends requests here, and the backend will use
our existing retrieval models to search through papers and
paragraphs and return the results as JSON.
"""

from fastapi import FastAPI
import json
from pathlib import Path
from src.index.indexer import load_index
from src.model.bm25 import BM25Model
from src.preprocessing.tokenizer import Tokenizer

app = FastAPI(title="IR Rangers Search API")

# So this finds the root foder of our project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File locations for our search engine
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "paper_index.json"
METADATA_PATH = PROCESSED_DIR / "paper_metadata.jsonl"
CONFIG_PATH = PROCESSED_DIR / "preprocessing_config.yaml"

# Same processing settings as our indexed papers
tokenizer = Tokenizer(config_path=CONFIG_PATH)

# Exsiting paper index
paper_index = load_index(INDEX_PATH)

# Create BM25 model with our indexed papers
bm25_model = BM25Model(paper_index, k1=0.9, b=0.4)

# Here we're storing paper information by doc ID. The goal here is to make it a quicker lookup.
paper_metadata = {}

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    for line in file:
        paper = json.loads(line)
        paper_metadata[paper["doc_id"]] = paper

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/search")
def search(q: str):
    tokens = tokenizer.tokenize(q)
    matches = bm25_model.search(tokens, k=10)
    results = []

    for doc_id, score in matches:
        paper = paper_metadata.get(doc_id)

        if paper:
            results.append({
                "id": doc_id,
                "type": "Paper",
                "via": "from",
                "source": "arXiv",
                "year": paper["year"],
                "format": "Research Paper",
                "title": paper["title"],
                "excerpt": paper["abstract"][:300]
            })

    return {
        "query": q,
        "results": results
    }

@app.get("/api/vocabulary")
def vocabulary(q: str):
    tokens = tokenizer.tokenize(q)
    unique_tokens = list(dict.fromkeys(tokens))

    return [
        {
            "id": "query",
            "label": "Search terms",
            "color": "yellow",
            "terms": [
                {"term": token, "on": True}
                for token in unique_tokens
            ]
        }
    ]

@app.get("/api/suggest")
def suggest(q: str):
    # Find paper titles
    query = q.strip().lower()

    if not query:
        return []

    suggestions = []

    for paper in paper_metadata.values():
        title = paper["title"]

        if title and title.lower().startswith(query):
            suggestions.append(title)

        if len(suggestions) == 5:
            break

    return suggestions