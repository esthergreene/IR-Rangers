"""
Writing retrieval results as TREC run files.

Each line has six tab-separated columns:

    qid  Q0  doc_id  rank  score  run_id

Ranks start at 1. Evalution is automated, so file names must match 
needed format exactly; These are built with run_filename().

Usage:
    from src.common.trec import run_filename, write_run

    path = Path("results") / run_filename("paper", "bm25", "study")
    write_run(path, results, run_id="bm25")
"""
from pathlib import Path

TEAM_NAME = "IRRangers" # TODO: Confirm if we're using this name for trec files

COLLECTIONS = {"paper", "paragraph"}
QUERY_SETS = {"train", "study", "test"} # only study and test runs are submitted

def run_filename(collection, model, query_set):
    """
    Build a run file name in the format the project requires.

    Args:
        collection (str): "paper" or "paragraph".
        model (str): Model name, e.g., "bm25" or "tfidf".
        query_set (str): "train", "study", or "test".
    
    Returns:
        str: e.g., "IRRangers_paper_bm25_study.tsv"
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"collection is {collection!r}; expected one of {sorted(COLLECTIONS)}.")
    if query_set not in QUERY_SETS:
        raise ValueError(f"query_set is {query_set!r}; expected one of {sorted(QUERY_SETS)}.")
    return f"{TEAM_NAME}_{collection}_{model}_{query_set}.tsv"

def write_run(path, results, run_id, digits=6):
    """
    Write ranked results to a TREC run file.

    Args:
        path (str or Path): Output file. Parent folders are created if needed.
        results (dict): query_id -> list of (doc_id, score), highest score first,
            as returned by a model's search(). Queries are written in the dict's order.
        run_id (str): Value for the last column. Must not contain whitespace.
        digits (int): Decimal places for score (degree of precision).

    Returns:
        int: Number of lines written.
    
    Raises:
        ValueError: If run_id contains whitespace, or a query's results are not sorted
            by score.
    """
    if not run_id or any(char.isspace() for char in run_id):
        raise ValueError(f"run_id must be non-empty with no whitespace; got {run_id!r}")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    with open(path, "w", encoding="utf-8") as f:
        for query_id, ranked in results.items():
            previous_score = float("inf")
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                if score > previous_score:
                    raise ValueError(f"Results for query {query_id} are not sorted by score.")
                previous_score = score

                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.{digits}f}\t{run_id}\n")
                num_lines += 1
    return num_lines