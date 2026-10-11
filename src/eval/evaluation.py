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
    report = compare(
        qrels=qrels, 
        runs=[run_bm25, run_tfidf], 
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

    # 2. Load the 4 runs ({paper, paragraph} x {bm25, tfidf})
    run_paper_bm25 = Run.from_file(RUNS_DIR / f"paper_bm25_{qset}.tsv", kind="trec")
    run_paper_tfidf = Run.from_file(RUNS_DIR / f"paper_tfidf_{qset}.tsv", kind="trec")
    run_paragraph_bm25 = Run.from_file(RUNS_DIR / f"paragraph_bm25_{qset}.tsv", kind="trec")
    run_paragraph_tfidf = Run.from_file(RUNS_DIR / f"paragraph_tfidf_{qset}.tsv", kind="trec")

    # 3. Evaluate Paper Runs
    print("\n--- PAPER COLLECTION EVALUATION ---")
    print("Graded Evaluation (Paper):")
    print(evaluate_graded(qrels_paper_graded, run_paper_bm25))
    print(evaluate_graded(qrels_paper_graded, run_paper_tfidf))
    
    print("\nStrict Evaluation (Paper):")
    print(evaluate_strict(qrels_paper_strict, run_paper_bm25))
    print(evaluate_strict(qrels_paper_strict, run_paper_tfidf))

    # 4. Evaluate Paragraph Runs
    print("\n--- PARAGRAPH COLLECTION EVALUATION ---")
    print("Graded Evaluation (Paragraph):")
    print(evaluate_graded(qrels_paragraph_graded, run_paragraph_bm25))
    print(evaluate_graded(qrels_paragraph_graded, run_paragraph_tfidf))
    
    print("\nStrict Evaluation (Paragraph):")
    print(evaluate_strict(qrels_paragraph_strict, run_paragraph_bm25))
    print(evaluate_strict(qrels_paragraph_strict, run_paragraph_tfidf))

    # 5. Run Statistical Significance Comparisons (BM25 vs TF-IDF)
    run_significance_test(
        qrels=qrels_paper_graded,
        run_bm25=run_paper_bm25,
        run_tfidf=run_paper_tfidf,
        metrics=["mrr", "ndcg@10", "precision@5"],
        title="Paper Models Significance Test (BM25 vs TF-IDF)"
    )

    run_significance_test(
        qrels=qrels_paragraph_graded,
        run_bm25=run_paragraph_bm25,
        run_tfidf=run_paragraph_tfidf,
        metrics=["mrr", "ndcg@10", "precision@5"],
        title="Paragraph Models Significance Test (BM25 vs TF-IDF)"
    )


if __name__ == "__main__":
    main()