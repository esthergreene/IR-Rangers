"""
Run a retrieval model over a query file and write a TREC-formatted run file.

BM25 and TF-IDF (Part 1) search their own paper and paragraph indexes. The 
neural models (Part 2) only index paragraphs, so for --collection paper they 
retrieve --paragraph-depth paragraphs and rank each paper by its best 
paragraph (MaxP, see src/common/maxp.py).

Usage, from the repository root:
    python scripts/run_retrieval.py --collection paper --queries study
    python scripts/run_retrieval.py --collection paragraph --queries study \
        --processed data/processed_sample --limit-queries 20
    
    python scripts/run_retrieval.py --collection paper --queries study --model sparse 
    python scripts/run_retrieval.py --collection paragraph --queries study --model sparse \
        --checkpoint checkpoints/sparse/run2/best_model  \
        --index data/processed_neural/sparse_paragraph_index_run2.npz
    python scripts/run_retrieval.py --collection paper --queries study --model sparse \
        --run-name sparse_zeroshot
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.bm25_config import load_bm25_params
from src.common.maxp import load_paper_of, maxp
from src.common.queries import load_queries
from src.common.trec import run_filename, write_run
from src.index.indexer import load_index as load_saved_index
from src.model.bm25 import BM25Model
from src.model.tfidf import TFIDFModel
from src.preprocessing.tokenizer import Tokenizer, clean_for_encoder, load_cleaning_config

RAW_DIR = REPO_ROOT / "data" / "raw"
QUERY_FILES = {"train": "Train.json", "study": "Study.json", "test": "Test.json"}
COLLECTION_FILES = {"paper": "paper_index.json", "paragraph": "paragraph_index.json"}

BAG_OF_WORDS_MODELS = ["bm25", "tfidf"]
NEURAL_MODELS = ["dense", "sparse"]
# Default paragraph index for each neural model, inside --neural-processed.
NEURAL_INDEX_FILES = {
    "sparse": "sparse_paragraph_index.npz",
    "dense": "dense_paragraph_index.npy",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run a retrieval model and write a TREC run file.")
    parser.add_argument("--collection", choices=sorted(COLLECTION_FILES), required=True, help="Collection to use for retrieval")
    parser.add_argument("--queries", choices=sorted(QUERY_FILES), required=True, help="Query set to use for retrieval")
    parser.add_argument("--model", choices=BAG_OF_WORDS_MODELS + NEURAL_MODELS, default="bm25", help="Retrieval model to use (default: bm25)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs", help="Folder to write the run file to (default: runs/)")
    parser.add_argument("--k", type=int, default=100, help="Results per query (default: 100)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Only run the first N queries (for quick tests)")
    parser.add_argument("--run-name", default=None, help="Model name for the run file name and run_id (default: --model), e.g. sparse_zeroshot")

    bow = parser.add_argument_group("BM25 / TF-IDF")
    bow.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed", help="Folder containing the saved indexes")
    bow.add_argument("--k1", type=float, default=None, help="BM25 k1 parameter (default: tuned value from config/bm25.yaml)")
    bow.add_argument("--b", type=float, default=None, help="BM25 b parameter (default: tuned value from config/bm25.yaml)")

    neural = parser.add_argument_group("dense / sparse")
    neural.add_argument("--neural-processed", type=Path, default=REPO_ROOT / "data" / "processed_neural", help="Folder with the cleaned paragraphs and neural indexes")
    neural.add_argument("--index", type=Path, default=None, help="Paragraph index file (default: the model's index in --neural-processed)")
    neural.add_argument("--checkpoint", default=None, help="Encoder name or checkpoint folder (default: the searcher's own default)")
    neural.add_argument("--paragraph-depth", type=int, default=2000, help="Paragraphs retrieved per query before MaxP, paper collection only (default: 2000)")

    args = parser.parse_args()
    if args.model in NEURAL_MODELS and args.collection == "paper" and args.paragraph_depth < args.k:
        parser.error("--paragraph-depth must be at least --k")
    return args

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

def build_bag_of_words_search(args):
    """
    Return search(query_text) -> [(doc_id, score)] for BM25 or TF-IDF.

    Queries are tokenized with the config saved alongisde the collections, 
    so they are processed exactly how the documents were.
    """
    tokenizer = Tokenizer(config_path=args.processed / "preprocessing_config.yaml")

    print(f"Loading {args.collection} index from {args.processed}...")
    index = load_index(args.collection, args.processed)
    model = build_model(args, index)
    print(f"  {index.num_docs:,} documents")

    def search(query_text):
        return model.search(tokenizer.tokenize(query_text), k=args.k)
    return search

def build_neural_searcher(args):
    """
    Load a neural searcher. Each one has search(query_text, k) returning 
    (paragraph_id, score) pairs, best first.

    Imported here so BM25/TF-IDF runs don't need torch.
    """
    index_path = args.index or args.neural_processed / NEURAL_INDEX_FILES[args.model]
    options = {} if args.checkpoint is None else {"model_name": args.checkpoint}

    print(f"Loading {args.model} model and paragraph index from {index_path}...")
    if args.model == "sparse":
        from src.model.sparse import SparseSearcher
        searcher = SparseSearcher(index_path=index_path, **options)
    else:
        from src.model.dense import DenseSearcher
        searcher = DenseSearcher(index_path=index_path, **options)
    print(f"  encoder: {args.checkpoint or 'searcher default'}")
    return searcher

def build_neural_search(args):
    """
    Return search(query_text) -> [(doc_id, score)] for a dense or sparse model.

    Queries get the same cleaning as the paragraphs (from the config saved in 
    --neural-processed). For the paper collection, papers are ranked by their 
    best paragraph among the top --paragraph-depth paragraphs (MaxP).
    """
    cleaning = load_cleaning_config(args.neural_processed / "preprocessing_config.yaml")
    searcher = build_neural_searcher(args)

    if args.collection == "paragraph":
        def search(query_text):
            return searcher.search(clean_for_encoder(query_text, **cleaning), k=args.k)
        return search

    paragraphs_path = args.neural_processed / "paragraphs.jsonl"
    print(f"Loading paragraph -> paper map from {paragraphs_path}...")
    paper_of = load_paper_of(paragraphs_path)
    print(f"  papers ranked by their best paragraph (MaxP) in the top {args.paragraph_depth:,}")

    def search(query_text):
        paragraphs = searcher.search(clean_for_encoder(query_text, **cleaning), k=args.paragraph_depth)
        return maxp(paragraphs, paper_of, k=args.k)
    return

def main():
    args = parse_args()
    run_name = args.run_name or args.__module__

    queries = load_queries(RAW_DIR / QUERY_FILES[args.queries])
    if args.limit_queries:
        queries = queries[: args.limit_queries]

    start = time.perf_counter()
    if args.model in NEURAL_MODELS:
        search = build_neural_search(args)
    else:
        search = build_bag_of_words_search(args)
    print(f"  ready in {time.perf_counter() - start:.1f}s")
    
    results = {}
    search_seconds = 0.0
    for query in queries:
        # Timed: query processing (tokenizing or encoding) and search.
        start = time.perf_counter()
        results[query.query_id] = search(query.text)
        search_seconds += time.perf_counter() - start
    
    path = args.output / run_filename(args.collection, args.model, args.queries)
    num_lines = write_run(path, results, run_id=args.model)

    print(f"Wrote {num_lines:,} lines for {len(queries):,} queries to {path}")
    print(f"Average search time: {1000 * search_seconds / len(queries):.1f} ms per query")
    
    no_results = sum(1 for ranked in results.values() if not ranked)
    if no_results:
        print(f"{no_results} queries returned no results")
    short = sum(1 for ranked in results.values() if 0 < len(ranked) < args.k)
    if short:
        hint = " (raise --paragraph-depth)" if args.model in NEURAL_MODELS and args.collection == "paper" else ""
        print(f"{short} queries returned fewer than {args.k} results {hint}")
        

if __name__ == "__main__":
    main()