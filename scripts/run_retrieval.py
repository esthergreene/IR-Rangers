"""
Run a retrieval model over a query file and write a TREC-formatted run file.

Usage, from the repository root:
    python scripts/run_retrieval.py --collection paper --queries study
    python scripts/run_retrieval.py --collection paragraph --queries study \
        --processed data/processed_sample --limit-queries 20
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.bm25_config import load_bm25_params
from src.common.queries import load_queries
from src.common.trec import run_filename, write_run
from src.index.indexer import load_index as load_saved_index
from src.model.bm25 import BM25Model
from src.model.tfidf import TFIDFModel
from src.preprocessing.tokenizer import Tokenizer

RAW_DIR = REPO_ROOT / "data" / "raw"
QUERY_FILES = {"train": "Train.json", "study": "Study.json", "test": "Test.json"}
COLLECTION_FILES = {"paper": "paper_index.json", "paragraph": "paragraph_index.json"}

def parse_args():
    parser = argparse.ArgumentParser(description="Run a retrieval model and write a TREC run file.")
    parser.add_argument("--collection", choices=sorted(COLLECTION_FILES), required=True, help="Collection to use for retrieval")
    parser.add_argument("--queries", choices=sorted(QUERY_FILES), required=True, help="Query set to use for retrieval")
    parser.add_argument("--model", choices=["bm25", "tfidf"], default="bm25", help="Retrieval model to use (default: bm25)")
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed", help="Folder containing the saved indexes")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs", help="Folder to write the run file to (default: runs/)")
    parser.add_argument("--k", type=int, default=100, help="Results per query (default: 100)")
    parser.add_argument("--k1", type=float, default=None, help="BM25 k1 parameter (default: tuned value from config/bm25.yaml)")
    parser.add_argument("--b", type=float, default=None, help="BM25 b parameter (default: tuned value from config/bm25.yaml)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Only run the first N queries (for quick tests)")

    return parser.parse_args()

def load_index(collection, processed_dir):
    """
    Load the index for one collection.
    """
    return load_saved_index(processed_dir / COLLECTION_FILES[collection])

def build_model(args, index):
    """
    Create the requested model.

    For BM25, any of k1/b not given on the command line comes from
    config/bm25.yaml for the chosen collection.
    """
    if args.model == "tfidf":
        print("  model: TF-IDF")
        return TFIDFModel(index)

    k1, b = args.k1, args.b
    if k1 is None or b is None:
        tuned_k1, tuned_b = load_bm25_params(args.collection)
        k1 = tuned_k1 if k1 is None else k1
        b = tuned_b if b is None else b
    source = "tuned" if args.k1 is None and args.b is None else "overridden on command line"
    print(f"  model: BM25 (k1={k1}, b={b}, {source})")
    return BM25Model(index, k1=k1, b=b)

def main():
    args = parse_args()

    # Tokenize queries with the config saved alongside the collections,
    # so queries are processed exactly how the documents were.
    tokenizer = Tokenizer(config_path=args.processed / "preprocessing_config.yaml")

    queries = load_queries(RAW_DIR / QUERY_FILES[args.queries])
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    
    print(f"Loading {args.collection} index from {args.processed}...")
    start = time.perf_counter()
    index = load_index(args.collection, args.processed)
    model = build_model(args, index)
    
    print(f"  {index.num_docs:,} documents, ready in {time.perf_counter() - start:.1f}s")

    results = {}
    search_seconds = 0.0
    for query in queries:
        tokens = tokenizer.tokenize(query.text)
        # Timed: Search only, not query tokenization or other overhead.
        start = time.perf_counter()
        results[query.query_id] = model.search(tokens, k=args.k)
        search_seconds += time.perf_counter() - start
    
    path = args.output / run_filename(args.collection, args.model, args.queries)
    num_lines = write_run(path, results, run_id=args.model)

    no_results = sum(1 for ranked in results.values() if not ranked)
    print(f"Wrote {num_lines:,} lines for {len(queries):,} queries to {path}")
    print(f"Average search time: {1000 * search_seconds / len(queries):.1f} ms per query")
    if no_results:
        print(f"{no_results} queries returned no results")

if __name__ == "__main__":
    main()