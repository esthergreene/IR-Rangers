"""
Evaluation script for retrieval outputs.
Takes TREC-formatted run files (qid, Q0, docid, rank, score, runid) 
found in runs/ and evaluates them against qrels using ranx.
Supports graded and strict metrics, as well as statistical significance testing.
"""

from ranx import Qrels, Run, evaluate, compare
import argparse
import sys
from pathlib import Path
from src.common.trec import run_filename

# Repository root setup to avoid absolute filesystem root bugs
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS_DIR = REPO_ROOT / "runs"


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
    Load a run file as a new Run object.

    ranx's make_comparable=True changes a Run in place (dropping queries that
    are not in the QRELs), so each evaluation gets its own fresh copy.
    """
    run = Run.from_file(str(RUNS_DIR / run_filename(collection, model, qset)), kind="trec")
    # Name used as the row label in the significance tables
    run.name = "BM25" if model == "bm25" else "TF-IDF"
    return run


def evaluate_graded(qrels, run):
    """Evaluates metrics using graded relevance (includes levels 1 and 2)."""
    # Note: Do not call set_relevance_level here so it uses all graded levels (0/1/2)
    metrics = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]
    return evaluate(qrels, run, metrics, make_comparable=True)


def evaluate_strict(qrels_strict, run):
    """Evaluates metrics using strict relevance (level 2 only)."""
    metrics = ["precision@5", "mrr"]
    return evaluate(qrels_strict, run, metrics, make_comparable=True)


def run_significance_test(qrels, run_bm25, run_tfidf, metrics, title):
    """Performs statistical significance testing (Student's t-test) between two models."""
    print(f"\n================ {title} ================")
    # compare() requires runs and QRELs to cover the same queries (e.g. strict
    # QRELs omit queries with no grade-2 document), so align each run first.
    runs = []
    for run in (run_bm25, run_tfidf):
        name = run.name
        run = run.make_comparable(qrels)
        run.name = name
        runs.append(run)
    report = compare(
        qrels=qrels, 
        runs=runs, 
        metrics=metrics, 
        max_p = 0.05,
        stat_test="student"
    )
    print(report)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval outputs against qrels.")
    parser.add_argument(
        "--queries", 
        choices=["train", "study"], 
        required=True, 
        help="Query set to evaluate (train or study; test qrels are withheld)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    qset = args.queries 
    
    # 1. Load Qrels for paper and paragraph (create separate instances for graded vs strict)
    qrels_paper_path = RAW_DIR / f"{qset.capitalize()}_paper_qrel.tsv"
    qrels_paragraph_path = RAW_DIR / f"{qset.capitalize()}_paragraph_qrel.tsv"
    
    # Graded Qrels (keeps all relevance grades)
    qrels_paper_graded = Qrels.from_file(qrels_paper_path, kind="trec")
    qrels_paragraph_graded = Qrels.from_file(qrels_paragraph_path, kind="trec")
    
    # Strict Qrels contain only grade-2 judgments and omit queries without them.
    qrels_paper_strict = load_strict_qrels(qrels_paper_path)
    qrels_paragraph_strict = load_strict_qrels(qrels_paragraph_path)

    # 2. Runs ({paper, paragraph} x {bm25, tfidf}) are loaded fresh with
    #    load_run() at each use, since evaluation can modify a Run in place.

    # 3. Evaluate Paper Runs
    print("\n--- PAPER COLLECTION EVALUATION ---")
    print("Graded Evaluation (Paper):")
    print(evaluate_graded(qrels_paper_graded, load_run("paper", "bm25", qset)))
    print(evaluate_graded(qrels_paper_graded, load_run("paper", "tfidf", qset)))
    
    print("\nStrict Evaluation (Paper):")
    print(evaluate_strict(qrels_paper_strict, load_run("paper", "bm25", qset)))
    print(evaluate_strict(qrels_paper_strict, load_run("paper", "tfidf", qset)))

    # 4. Evaluate Paragraph Runs
    print("\n--- PARAGRAPH COLLECTION EVALUATION ---")
    print("Graded Evaluation (Paragraph):")
    print(evaluate_graded(qrels_paragraph_graded, load_run("paragraph", "bm25", qset)))
    print(evaluate_graded(qrels_paragraph_graded, load_run("paragraph", "tfidf", qset)))
    
    print("\nStrict Evaluation (Paragraph):")
    print(evaluate_strict(qrels_paragraph_strict, load_run("paragraph", "bm25", qset)))
    print(evaluate_strict(qrels_paragraph_strict, load_run("paragraph", "tfidf", qset)))

    # 5. Run Statistical Significance Comparisons (BM25 vs TF-IDF)
    run_significance_test(
        qrels=qrels_paper_graded,
        run_bm25=load_run("paper", "bm25", qset),
        run_tfidf=load_run("paper", "tfidf", qset),
        metrics=["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"],
        title="Paper Models Significance Test (BM25 vs TF-IDF)"
    )

    run_significance_test(
        qrels=qrels_paper_strict,
        run_bm25=load_run("paper", "bm25", qset),
        run_tfidf=load_run("paper", "tfidf", qset),
        metrics=["mrr", "precision@5"],
        title="Paper Models Significance Test, strict (BM25 vs TF-IDF)"
    )

    run_significance_test(
        qrels=qrels_paragraph_graded,
        run_bm25=load_run("paragraph", "bm25", qset),
        run_tfidf=load_run("paragraph", "tfidf", qset),
        metrics=["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"],
        title="Paragraph Models Significance Test (BM25 vs TF-IDF)"
    )

    run_significance_test(
        qrels=qrels_paragraph_strict,
        run_bm25=load_run("paragraph", "bm25", qset),
        run_tfidf=load_run("paragraph", "tfidf", qset),
        metrics=["mrr", "precision@5"],
        title="Paragraph Models Significance Test, strict (BM25 vs TF-IDF)"
    )


if __name__ == "__main__":
    main()