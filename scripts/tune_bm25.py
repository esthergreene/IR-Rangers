"""
Tune BM25's k1 and b on a sample of the Train queries.

For one collection, runs BM25 over a grid of (k1, b) values on a random but
reproducible sample of Train queries, and scores each setting with a single
metric. Part of the sample is held out: the best setting is chosen based on
the tuning part only, then checked on the held-out part to see whether it
generalizes. Every grid result is saved as to a CSV.

Study queries are never used here, since they are reserved for reporting.

Usage, from the repository root:
    python scripts/tune_bm25.py --collection paper
    python scripts/tune_bm25.py --collection paragraph --num-queries 200 \
        --k1 0.6 0.9 1.2 --b 0.4 0.6 0.8
"""
import argparse
import csv
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from ranx import Qrels, Run, evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.queries import load_queries
from src.index.indexer import load_index
from src.model.bm25 import BM25Model
from src.preprocessing.tokenizer import Tokenizer

RAW_DIR = REPO_ROOT / "data" / "raw"
INDEX_FILES = {"paper": "paper_index.json", "paragraph": "paragraph_index.json"}
QREL_FILES = {"paper": "Train_paper_qrel.tsv", "paragraph": "Train_paragraph_qrel.tsv"}

def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search BM25's k1 and b on Train queries.")
    parser.add_argument("--collection", choices=sorted(INDEX_FILES), required=True)
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed", 
                        help="Folder with the saved indexes and preprocessing config")
    parser.add_argument("--num-queries", type=int, default=400, 
                        help="How many Train queries to sample (default: 400)")
    parser.add_argument("--dev-fraction", type=float, default=0.2,
                        help="Share of the sample held out for checking the best setting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the query sample")
    parser.add_argument("--k1", type=float, nargs="+", default=[0.6, 0.9, 1.2, 1.5, 1.8])
    parser.add_argument("--b", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.75, 0.9])
    parser.add_argument("--metric", default="ndcg@5", help="ranx metric to optimize (default: ndcg@5)")
    parser.add_argument("--k", type=int, default=100, help="Results per query")
    parser.add_argument("--output", type=Path, default=None, help="CSV file for the grid (default: results/bm25_tuning_<collection>.csv)")

    return parser.parse_args()

def read_qrels(path):
    """Return query_id -> {doc_id: relevance} from a TREC QREL file."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, relevance = line.split()
            qrels[query_id][doc_id] = int(relevance)
    return dict(qrels)

def run_queries(model, tokenized_queries, k):
    """
    Search every query and return a ranx-ready run: query_id -> {doc_id: score}.

    Queries with no results are left out; make_comparable in score_run() then
    counts them as 0.
    """
    run = {}
    for query_id, tokens in tokenized_queries.items():
        ranked = model.search(tokens, k=k)
        if ranked:
            run[query_id] = dict(ranked)
    return run

def score_run(qrels, run, metric):
    """Mean value of one metric over the queries in qrels."""
    return evaluate(Qrels(qrels), Run(run), metric, make_comparable=True)

def main():
    args = parse_args()
    output = args.output or REPO_ROOT / "results" / f"bm25_tuning_{args.collection}.csv"

    # Same tokenizer settings the collections were built with
    tokenizer = Tokenizer(config_path=args.processed / "preprocessing_config.yaml")
    qrels = read_qrels(RAW_DIR / QREL_FILES[args.collection])

    # Sample judged Train queries reproducibly, then split into tunring and held-out parts
    queries = [query for query in load_queries(RAW_DIR / "Train.json") if query.query_id in qrels]
    sample = random.Random(args.seed).sample(queries, min(args.num_queries, len(queries)))
    num_dev = int(len(sample) * args.dev_fraction)
    dev_queries, tune_queries = sample[:num_dev], sample[num_dev:]

    # Tokenize once; only the BM25 parameters change between grid cells
    tune_tokens = {query.query_id: tokenizer.tokenize(query.text) for query in tune_queries}
    dev_tokens = {query.query_id: tokenizer.tokenize(query.text) for query in dev_queries}
    tune_qrels = {query_id: qrels[query_id] for query_id in tune_tokens}
    dev_qrels = {query_id: qrels[query_id] for query_id in dev_tokens}

    print(f"{args.collection}: {len(tune_queries)} tuning queries, {len(dev_queries)} held out "
            f"(seed {args.seed}), optimizing {args.metric}")
    print(f"Loading {args.collection} index...")
    start = time.perf_counter()
    index = load_index(args.processed / INDEX_FILES[args.collection])

    # IDF is computed once here; set_params() only recomputes length normalization
    model = BM25Model(index, k1=args.k1[0], b=args.b[0])
    print(f"    ready in {time.perf_counter() - start:.1f}s")

    grid = list(itertools.product(args.k1, args.b))
    rows = []
    grid_start = time.perf_counter()
    for cell, (k1, b) in enumerate(grid, start=1):
        model.set_params(k1, b)
        cell_start = time.perf_counter()
        value = score_run(tune_qrels, run_queries(model, tune_tokens, args.k), args.metric)
        seconds = time.perf_counter() - cell_start

        rows.append({"k1": k1, "b": b, args.metric: value, "seconds": round(seconds, 1)})
        remaining = (time.perf_counter() - grid_start) / cell * (len(grid) - cell)
        print(f"[{cell}/{len(grid)}] k1={k1:<4} b={b:<4} {args.metric}={value:.4f} "
              f"({seconds:.0f}s, about {remaining / 60:.0f} min left)")
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k1", "b", args.metric, "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved the grid to {output}")
    
    # On ties, max() keeps the first setting in grid order.
    best = max(rows, key=lambda row: row[args.metric])
    print(f"Best on tuning queries: k1={best['k1']}, b={best['b']}, "
          f"{args.metric}={best[args.metric]:.4f}")
    
    if dev_queries:
        print(f"\nHeld-out check ({len(dev_queries)} queries, {args.metric}):")
        for label, (k1, b) in [("best", (best["k1"], best["b"])),
                               ("Anserini default", (0.9, 0.4)),
                               ("textbook", (1.2, 0.75))]:
            model.set_params(k1, b)
            value = score_run(dev_qrels, run_queries(model, dev_tokens, args.k), args.metric)
            print(f"    {label:<17} k1={k1:<4} b={b:<4} {value:.4f}")
    
    name = args.collection.upper()
    print(f"\nFor scripts/run.sh:\n   BM25_{name}_K1={best['k1']}\n   BM25_{name}_B={best['b']}")

if __name__ == "__main__":
    main()