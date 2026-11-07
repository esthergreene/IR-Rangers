"""
Evaluate TREC run files in runs/ against the QRELs with ranx.

For each collection (paper, paragraph), prints one table per relevance 
setting, with every model's scores and paired significance tests between 
every pair of models:

    graded  nDCG, nDCG@5, MRR, P@5, P@10, Recall@100
        (grades 1 and 2 count as relevant; nDCG uses the grades)
    strict MRR, P@5
        (only grade 2 counts; queries without a grade-2 document are left out)

In each table, a superscript letter on a score means that model is significantly 
better (p < 0.05) than the model with that letter.

Usage, from the repository root:
    python -m src.eval.evaluation --queries study
    python -m src.eval.evaluation --queries study --models bm25 sparse_zeroshot sparse
    python -m src.eval.evaluation --queries study --models bm25 tfidf    # Part 1 Comparison
    python -m src.eval.evaluation --queries study --latex results/latex
"""
import argparse
from pathlib import Path

from ranx import Qrels, Run, compare, fuse

from src.common.trec import run_filename

# Repository root setup to avoid absolute filesystem root bugs
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS_DIR = REPO_ROOT / "runs"

COLLECTIONS = ["paper", "paragraph"]
DEFAULT_MODELS = ["bm25", "dense", "sparse"] # BM25 first, so it gets letter "a" in the tables
GRADED_METRICS = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10", "recall@100"]
STRICT_METRICS = ["mrr", "precision@5"]
DISPLAY_NAMES = {"bm25": "BM25", "tfidf": "TF-IDF", "dense": "Dense", "sparse": "SPLADE"}

RRF_PREFIX = "rrf:"

def display_name(model):
    """Table label for a run name, e.g., rrf:bm25+sparse -> RRF(BM25+SPLADE)."""
    if model.startswith(RRF_PREFIX):
        parts = model[len(RRF_PREFIX):].split("+")
        return "RRF(" + "+".join(DISPLAY_NAMES.get(part, part) for part in parts) + ")"
    return DISPLAY_NAMES.get(model, model)

def fuse_rrf(collection, models, qset, k=60, depth=100):
    """
    Fuse runs with Reciprocal Rank Fusion: score = sum of 1 / (k + rank).

    k = 60 is the standard value (Cormack et al., 2009) and is not tuned.
    The fused run keeps only the top `depth` per query, like every other run, 
    so nDCG and Recall@100 aren't computed over up to 200 documents.
    """
    runs = [load_run(collection, model, qset) for model in models]
    fused = fuse(runs=runs, method="rrf", params={"k": k})
    top = {
        query_id: dict(sorted(scores.items(), key=lambda item: -item[1])[:depth])
        for query_id, scores in fused.to_dict().items()
    }
    return Run(top)


def load_strict_qrels(path):
    """
    Load QRELs with only fully relevant (grade 2) judgments.

    Queries with no grade-2 document are omitted, so strict metrics are
    averaged only over queries that have something to find.
    """
    strict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, relevance = line.split()
            if int(relevance) == 2:
                strict.setdefault(query_id, {})[doc_id] = 2
    return Qrels(strict)


def load_run(collection, model, qset):
    """
    Load a run file (or fuse several, for rrf:a+b) as a new Run object.

    ranx's make_comparable() changes a Run in place (dropping queries that
    are not in the QRELs), so each table gets its own fresh copy.
    """
    if model.startswith(RRF_PREFIX):
        run = fuse_rrf(collection, model[len(RRF_PREFIX):].split("+"), qset)
    else:
        run = Run.from_file(str(RUNS_DIR / run_filename(collection, model, qset)), kind="trec")
    run.name = DISPLAY_NAMES.get(model, model)
    return run

def available_models(collection, models, qset):
    """Keep the models whose run file exists, warning about the rest."""
    found = []
    for model in models:
        parts = model[len(RRF_PREFIX):].split("+") if model.startswith(RRF_PREFIX) else [model]
        missing = [p for p in parts if not (RUNS_DIR / run_filename(collection, p, qset)).exists()]
        if missing:
            print(f"Skipping {model} for {collection}: no run file for {', '.join(missing)}")
        else:
            found.append(model)
    return found

def print_comparison(qrels, collection, models, qset, metrics, title, stat_test, latex_path=None):
    """
    Score every model and test every pair of models on the same queries.

    compare() needs the runs and QRELs to cover the same queries (strict QRELs 
    leave some queries out), so each run is aligned to the QRELs first.
    """
    print(f"\n================ {title} ================")
    runs = []
    for model in models:
        run = load_run(collection, model, qset)
        name = run.name
        run = run.make_comparable(qrels)
        run.name = name
        runs.append(run)

    report = compare(qrels=qrels, runs=runs, metrics=metrics, max_p=0.05, stat_test=stat_test)
    print(report)

    if latex_path is not None:
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(report.to_latex(), encoding="utf-8")
        print(f"(LaTeX table saved to {latex_path})")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval outputs against qrels.")
    parser.add_argument(
        "--queries", 
        choices=["train", "study"], 
        required=True, 
        help="Query set to evaluate (train or study; test qrels are withheld)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Run names to compare, as in the run file names (default: bm25 dense sparse)",
    )
    parser.add_argument(
        "--stat-test",
        choices=["student", "fisher", "tukey"],
        default="student",
        help="Significance test (default: student, paired t-test as in Part 1)",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="Folder to also save each table as LaTeX, for the paper",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    qset = args.queries 
    
    if qset == "train":
        print("Note: the neural models were fine-tuned on Train queries, so "
        "these are not held-out scores.")

    for collection in COLLECTIONS:
        models = available_models(collection, args.models, qset)
        if not models:
            continue

        qrels_path = RAW_DIR / f"{qset.capitalize()}_{collection}_qrel.tsv"
        qrels_graded = Qrels.from_file(str(qrels_path), kind="trec")
        qrels_strict = load_strict_qrels(qrels_path)

        print(f"\n--- {collection.upper()} COLLECTION ({qset}) ---")
        for setting, qrels, metrics in (
            ("graded", qrels_graded, GRADED_METRICS),
            ("strict", qrels_strict, STRICT_METRICS),
        ):
            latex_path = args.latex / f"{qset}_{collection}_{setting}.tex" if args.latex else None
            title = f"{collection.capitalize()}, {setting}" + (" (grade 2 only)" if setting == "strict" else "")
            print_comparison(qrels, collection, models, qset, metrics, title, args.stat_test, latex_path)


if __name__ == "__main__":
    main()