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

from src.common.queries import load_queries
from src.common.trec import run_filename, write_run
from src.index.indexer import CorpusIndex
from src.model.bm25 import BM25Model
from src.preprocessing.tokenizer import Tokenizer

RAW_DIR = REPO_ROOT / "data" / "raw"
QUERY_FILES = {"train": "Train.json", "study": "Study.json", "test": "Test.json"}
COLLECTION_FILES = {"paper": "papers.json", "paragraph": "paragraphs.json"}

def parse_args():
    parser = argparse.ArgumentParser(description="Run a retrieval model and write a TREC run file.")
    parser.add_argument("--collection", choices=sorted(COLLECTION_FILES), required=True, help="Collection to use for retrieval")
    parser.add_argument("--queries", choices=sorted(QUERY_FILES), required=True, help="Query set to use for retrieval")
    parser.add_argument("--model", choices=["bm25", "tfidf"], default="bm25", help="Retrieval model to use (default: bm25)")
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed", help="Folder written by build_collections.py")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs", help="Folder to write the run file to (default: runs/)")
    parser.add_argument("--k", type=int, default=100, help="Results per query (default: 100)")
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1 parameter (default: 0.9)")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b parameter (default: 0.4)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Only run the first N queries (for quick tests)")

    return parser.parse_args()

def load_index(collection, processed_dir):
    """
    Load the index for one collection.
    """
    return CorpusIndex.load_index(processed_dir / COLLECTION_FILES[collection])

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
    if args.model.lower() == "bm25":
        model = BM25Model(index, k1=args.k1, b=args.b)
    elif args.model.lower() == "tfidf":
        model = TFIDFModel(index)  # Fake name for now
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"  {index.num_docs:,} documents, ready in {time.perf_counter() - start:.1f}s")

    results = []
    search_seconds = 0.0
    for query in queries:
        tokens = tokenizer.tokenize(query["query_text"])
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