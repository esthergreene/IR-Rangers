"""
Run fine-tuned Sentence-BERT retrieval over our validation queries.

Searches the full paragraph FAISS index and saves results
in the team's TREC run format.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model.dense import DenseModel
from src.training.data import load_examples
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config
from src.common.trec import write_run

# Project directory stuff
VAL_PATH = REPO_ROOT / "data/processed_neural/training/val.jsonl"
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = REPO_ROOT / "data/dense_indexes/pretrained_paragraphs"
INDEX_PATH = INDEX_DIR / "paragraph_dense.faiss"
IDS_PATH = INDEX_DIR / "paragraph_dense_ids.json"
CLEANING_CONFIG = REPO_ROOT / "config/preprocessing_neural.yaml"
OUTPUT_PATH = REPO_ROOT / "runs/IRRangers_paragraph_sbert_pretrained_val.tsv"

def load_validation_queries():
    """
    Load the held-out validation queries and apply the same
    cleaning used during Sentence-BERT fine-tuning.
    """
    examples = load_examples(VAL_PATH)
    cleaning = load_cleaning_config(CLEANING_CONFIG)
    queries = {}

    for example in examples:
        query_id = str(example["query_id"])
        query_text = clean_for_encoder(example["query"], **cleaning)
        queries[query_id] = query_text

    print(f"Loaded {len(queries)} validation queries.")

    return queries

def run_dense_retrieval(queries, k=100):
    """
    Search the full paragraph collection using our
    fine-tuned Sentence-BERT model.
    """
    print("Loading pretrained Sentence-BERT model...")

    model = DenseModel(model_path=str(MODEL_PATH))

    print("Loading full paragraph FAISS index...")

    model.load_index(INDEX_PATH, IDS_PATH)

    print(f"Searching {len(queries)} validation queries...")

    results = {}

    for count, (query_id, query_text) in enumerate(queries.items(), start=1):
        results[query_id] = model.search(query_text, k=k)

        if count % 25 == 0:
            print(f"Completed {count}/{len(queries)} queries.")

    return results

def main():
    print("Starting pretrained Sentence-BERT retrieval...")

    queries = load_validation_queries()
    results = run_dense_retrieval(queries, k=100)
    num_lines = write_run(OUTPUT_PATH, results, run_id="sbert_pretrained")

    print(f"Saved {num_lines:,} results to {OUTPUT_PATH}")
    print("Dense retrieval complete!")

if __name__ == "__main__":
    main()
