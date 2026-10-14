"""
This file handles the backend API for our search engine.

The frontend sends requests here, and the backend will use
our existing retrieval models to search through papers and
paragraphs and return the results as JSON.
"""

from array import array
import json
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from src.common.bm25_config import load_bm25_params
from src.index.indexer import load_index
from src.model.bm25 import BM25Model
from src.model.tfidf import TFIDFModel
from src.preprocessing.tokenizer import Tokenizer

app = FastAPI(title="IR Rangers Search API")

# So this finds the root foder of our project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File locations for our search engine
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "paper_index.json"
PARAGRAPH_INDEX_PATH = PROCESSED_DIR / "paragraph_index.json"
METADATA_PATH = PROCESSED_DIR / "paper_metadata.jsonl"
PARAGRAPH_COLLECTION_PATH = PROCESSED_DIR / "paragraphs.jsonl"
CONFIG_PATH = PROCESSED_DIR / "preprocessing_config.yaml"

# Same processing settings as our indexed papers
tokenizer = Tokenizer(config_path=CONFIG_PATH)

# Existing paper index
paper_index = load_index(INDEX_PATH)
collection_indexes = {"paper": paper_index}
search_models = {
    ("bm25", "paper"): BM25Model(paper_index, *load_bm25_params("paper"))
}
paragraph_offsets = None

# Here we're storing paper information by doc ID. The goal here is to make it a quicker lookup.
paper_metadata = {}

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Paper metadata is missing. Run python scripts/build_paper_metadata.py first."
    )

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    for line in file:
        paper = json.loads(line)
        paper_metadata[paper["doc_id"]] = paper

searchable_titles = [
    (paper["title"].lower(), paper["title"])

    for paper in paper_metadata.values()

    if paper.get("title")
]

def get_collection_index(collection):
    if collection not in collection_indexes:
        index_path = {
            "paper": INDEX_PATH,
            "paragraph": PARAGRAPH_INDEX_PATH,
        }[collection]
        if not index_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"The {collection} search index is not available.",
            )
        collection_indexes[collection] = load_index(index_path)
    return collection_indexes[collection]

def get_search_model(model, collection):
    key = (model, collection)
    if key not in search_models:
        index = get_collection_index(collection)
        search_models[key] = (
            BM25Model(index, *load_bm25_params(collection))
            if model == "bm25"
            else TFIDFModel(index)
        )
    return search_models[key]

def get_paragraph_offsets(index):
    global paragraph_offsets
    if paragraph_offsets is None:
        if not PARAGRAPH_COLLECTION_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Paragraph collection text is not available.",
            )
        offsets = array("Q")
        with open(PARAGRAPH_COLLECTION_PATH, "rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                offsets.append(offset)
        if len(offsets) != index.num_docs:
            raise RuntimeError(
                "Paragraph collection and index contain different document counts."
            )
        paragraph_offsets = offsets
    return paragraph_offsets

def get_paragraph_metadata(doc_id, index):
    offsets = get_paragraph_offsets(index)
    offset = offsets[index.doc_number(doc_id)]
    with open(PARAGRAPH_COLLECTION_PATH, "rb") as file:
        file.seek(offset)
        paragraph = json.loads(file.readline())
    if paragraph["doc_id"] != doc_id:
        raise RuntimeError(f"Paragraph metadata did not match indexed document {doc_id}.")
    return paragraph

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": ["bm25", "tfidf"],
        "collections": ["paper", "paragraph"],
    }

# Results per query that can be browsed: the same depth as the submitted TREC run files.
MAX_RESULTS = 100

def make_excerpt(text, query, max_length=300):
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    query_tokens = set(tokenizer.tokenize(query))

    def relevance(sentence):
        sentence_tokens = set(tokenizer.tokenize(sentence))
        return len(query_tokens & sentence_tokens)

    best_sentence = max(sentences, key=relevance, default="")

    if relevance(best_sentence) == 0:
        best_sentence = sentences[0]

    if len(best_sentence) <= max_length:
        return best_sentence

    return best_sentence[:max_length].rsplit(" ", 1)[0] + "..."

@app.get("/api/search")
def search(
    q: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    model: Literal["bm25", "tfidf"] = "bm25",
    collection: Literal["paper", "paragraph"] = "paper",
):
    tokens = tokenizer.tokenize(q)

    # Rank the top MAX_RESULTS once, then return the requested page of them.
    ranked = get_search_model(model, collection).search(tokens, k=MAX_RESULTS)
    start = (page - 1) * page_size
    matches = ranked[start:start + page_size]
    results = []

    for doc_id, score in matches:
        if collection == "paper":
            paper = paper_metadata.get(doc_id)
            excerpt = make_excerpt(paper["abstract"], q) if paper else ""
        else:
            paragraph = get_paragraph_metadata(doc_id, get_collection_index("paragraph"))
            paper = paper_metadata.get(paragraph["paper_id"])
            excerpt = make_excerpt(paragraph["text"], q)

        if paper:
            results.append({
                "id": doc_id,
                "type": "Paper" if collection == "paper" else "Paragraph",
                "via": "from",
                "source": "arXiv",
                "year": paper["year"],
                "format": "Research Paper",
                "title": paper["title"],
                "excerpt": excerpt,
            })

    return {
        "query": q,
        "results": results,
        "total": len(ranked),
        "model": model,
        "collection": collection,
    }

@app.get("/api/document/{doc_id}")
def document_preview(
    doc_id: str,
    collection: Literal["paper", "paragraph"] = "paper",
):
    index = get_collection_index(collection)
    try:
        index.doc_number(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found.") from None

    if collection == "paper":
        paper = paper_metadata.get(doc_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Document preview is not available.")
        content = paper["abstract"]
        title = paper["title"]
    else:
        paragraph = get_paragraph_metadata(doc_id, index)
        paper = paper_metadata.get(paragraph["paper_id"])
        if paper is None:
            raise HTTPException(status_code=404, detail="Document preview is not available.")
        content = paragraph["text"]
        title = paper["title"]

    return {
        "id": doc_id,
        "title": title,
        "type": "Paper" if collection == "paper" else "Paragraph",
        "source": "arXiv",
        "year": paper["year"],
        "content": content,
    }

@app.get("/api/vocabulary")
def vocabulary(q: str):
    words = q.split()
    unique_tokens = list(dict.fromkeys(word for word in words if tokenizer.tokenize(word)))

    return [
        {
            "id": "query",
            "label": "Search terms",
            "color": "yellow",
            "terms": [
                {"term": token, "on": True}
                for token in unique_tokens
            ],
        }
    ]

@app.get("/api/suggest")
def suggest(q: str):
    # Find paper titles
    query = q.strip().lower()

    if not query:
        return []

    suggestions = []

    for lowercase_title, original_title in searchable_titles:
        if query in lowercase_title:
            suggestions.append(original_title)

        if len(suggestions) == 5:
            break

    return suggestions
