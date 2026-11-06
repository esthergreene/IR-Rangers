"""
Build training and validation data for fine-tuning the neural models.

Splits the judged Train queries into training and validation sets, then 
writes one example per query with its relevant paragraphs (positives) and 
non-relevant paragraphs (negatives). Only Train queries are used: Study and 
Test are reserved for evaluation and must never be used for training.

Negatives are BM25 hard negatives: paragraphs BM25 retrieves for the query
that are not judged relevant. They share words and topic with the query, so
they teach the model more than random paragraphs would. (The Train QRELs have
no grade-0 judgments, so there are no judged non-relevant paragraphs to use.)

    - The top --skip-top BM25 ranks are skipped, since an unjudged paragraph
      that high is often relevant but simply was not judged.
    - From the remaining ranks, --num-negatives paragraphs are sampled at
      random rather than taken in order, so each query gets a mix of close
      calls and easier negatives. Sampling is seeded per query, so the same
      settings always give the same negatives.

The split is by query, so no validation query appears in training, and is 
stratified by label (Easy/Medium/Hard) so both sets have the same mix.

Needs a BM25 run on the Train paragraphs first:
    python scripts/run_retrieval.py --collection paragraph --queries train

Writes to the output folder:
    train.jsonl     one example per training query
    val.jsonl       one example per validation query (for validation loss)
    manifest.json   settings and counts for this build

Each line looks like:
    {"query_id": "17", "query": "...", "label": "Easy", 
     "positives": {"2409.10230_35": 2, "2408.02148_9": 1}, 
     "negatives": ["2401.00012_4", ...]}

Usage, from the repository root:
    python scripts/build_training_data.py
    python scripts/build_training_data.py --val-fraction 0.15 --num-negatives 50
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.maxp import load_paper_of
from src.common.queries import load_queries
from src.common.trec import run_filename

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"

def parse_args():
    parser = argparse.ArgumentParser(description="Build training and validation " \
    "data from the Train queries.")
    parser.add_argument("--qrels", type=Path, default=RAW_DIR / "Train_paragraph_qrel.tsv", 
                        help="Train paragraph QREL file")
    parser.add_argument("--bm25-run", type=Path, default=REPO_ROOT / "runs" / run_filename("paragraph", "bm25", "train"),
                        help="BM25 run on the Train paragraphs, used for hard negatives")
    parser.add_argument("--paragraphs", type=Path, default=PROCESSED_DIR / "paragraphs.jsonl", 
                        help="Cleaned-text paragraph collection")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "training")
    parser.add_argument("--val-fraction", type=float, default=0.1, 
                        help="Share of queries held out for validation (default: 0.1)")
    parser.add_argument("--num-negatives", type=int, default=30, help="Negatives kept per query (default: 30)")
    parser.add_argument("--skip-top", type=int, default=20, help="BM25 ranks skipped before taking negatives (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split (default: 42)")
    return parser.parse_args()

def read_qrels(path):
    """Return query_id -> {doc_id: relevance} from a TREC QREL file."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, relevance = line.split()
            qrels[query_id][doc_id] = int(relevance)
    return dict(qrels)

def read_run(path):
    """Return query_id -> [doc_id, ...] in rank order from a TREC run file."""
    ranked = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, rank, _score, _run_id = line.split()
            ranked[query_id].append((int(rank), doc_id))
    return {query_id: [doc_id for _, doc_id in sorted(pairs)] for query_id, pairs in ranked.items()}

def split_by_label(queries, val_fraction, seed):
    """
    Split queries into training and validation sets, stratified by label.

    Each label group is shuffled and split on its own, so both sets have the 
    same mix of Easy, Medium, and Hard queries.

    Args:
        queries (list of Query): Judged Train queries.
        val_fraction (float): Share of each label group held out
        seed (int): Random seed, so the split is reproducible.
    
    Returns:
        tuple: (train_queries, val_queries), each a list of Query.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for query in queries:
        groups[query.label].append(query)

    train, val = [], []
    # Labels in a fixed order, so the seed alone decides the split
    for label in sorted(groups, key=str):
        group = groups[label]
        rng.shuffle(group)
        num_val = round(len(group) * val_fraction)
        val.extend(group[:num_val])
        train.extend(group[num_val:])
    return train, val

def build_example(query, judged, bm25_ranking, in_collection, num_negatives, skip_top, seed):
    """
    Build one training example, or return None if the query has no usable positive.

    Args:
        query (Query): The query.
        judged (dict): doc_id -> relevance for this query, from the QRELs.
        bm25_ranking (list): BM25's ranked doc_ids for this query.
        in_collection (container): IDs of paragraphs in the cleaned collection.
        num_negatives (int): Negatives to keep.
        skip_top (int): BM25 ranks to skip before taking negatives.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict or None: The example, in the format shown at the top of this file.
    """
    # Paragraphs removed by cleaning (e.g., math only) cannot be encoded.
    positives = {doc_id: grade for doc_id, grade in judged.items() if grade > 0 and doc_id in in_collection}
    if not positives:
        return None

    negatives = [doc_id for doc_id, grade in judged.items() if grade == 0 and doc_id in in_collection]
    # for doc_id in bm25_ranking[skip_top:]:
    #     if len(negatives) >= num_negatives:
    #         break
    #     if doc_id not in judged and doc_id in in_collection:
    #         negatives.append(doc_id)

    # Sample from below the top BM25 ranks rather than taking them in order, so 
    # negatives mix close calls with easier ones and fewer are unjudged relevant 
    # paragraphs. Seeded per query, so the same build gives the same negatives.
    candidates = [doc_id for doc_id in bm25_ranking[skip_top:] if doc_id not in judged and doc_id in in_collection]
    rng = random.Random(f"{seed}-{query.query_id}")
    negatives += rng.sample(candidates, min(num_negatives - len(negatives), len(candidates)))

    return{
        "query_id": query.query_id,
        "query": query.text,
        "label": query.label,
        "positives": positives,
        "negatives": negatives[:num_negatives]
    }

def main():
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        sys.exit("--val-fraction must be between 0 and 1.")
    if not args.bm25_run.exists():
        sys.exit(f"{args.bm25_run} not found. Run BM25 on the Train paragraphs first:\n"
                 "  python scripts/run_retrieval.py --collection paragraph --queries train")

    print("Building training data...")

    qrels = read_qrels(args.qrels)
    bm25 = read_run(args.bm25_run)
    in_collection = load_paper_of(args.paragraphs).keys() # Only the IDs are needed here

    # Only judged Train queries can be used for training
    queries = [query for query in load_queries(RAW_DIR / "Train.json") if query.query_id in qrels]
    train_queries, val_queries = split_by_label(queries, args.val_fraction, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in (("train", train_queries), ("val", val_queries)):
        examples, skipped = [], []
        for query in split:
            example = build_example(query, qrels[query.query_id], bm25.get(query.query_id, []), 
                                    in_collection, args.num_negatives, args.skip_top, args.seed)
            if example is None:
                skipped.append(query.query_id)
            else:
                examples.append(example)

        with open(args.output / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        counts[name] = {
            "queries": len(examples),
            "positives": sum(len(example["positives"]) for example in examples),
            "negatives": sum(len(example["negatives"]) for example in examples), 
            "labels": dict(Counter(str(example["label"]) for example in examples)),
            "skipped_no_positive": skipped,
        }

        print(f"{name}: {len(examples):,} queries, {counts[name]['positives']:,} positives, "
              f"{counts[name]['negatives']:,} negatives ({len(skipped)} skipped, no positive)")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qrels": str(args.qrels),
        "bm25_run": str(args.bm25_run),
        "paragraphs": str(args.paragraphs),
        "settings": {
            "val_fraction": args.val_fraction,
            "num_negatives": args.num_negatives,
            "skip_top": args.skip_top,
            "seed": args.seed,
        },
        "counts": counts,
    }
    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote training data to {args.output}")

if __name__ == "__main__":
    main()