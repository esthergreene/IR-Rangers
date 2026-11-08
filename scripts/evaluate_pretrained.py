"""
Evaluate fine-tuned Sentence-BERT retrieval resultsusing our 200 held-out validation queries.
"""

from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ranx import Qrels, Run, evaluate
from src.training.data import load_examples

# Project file paths
VAL_PATH = REPO_ROOT / "data/processed_neural/training/val.jsonl"
QRELS_PATH = Path(os.environ.get("IR_QRELS_PATH", str(Path.home() / "datasets/IR26/Train_paragraph_qrel.tsv")))
RUN_PATH = REPO_ROOT / "runs/IRRangers_paragraph_sbert_pretrained_val.tsv"

def load_validation_qrels():
    """
    Load the original relevance judgments and keep only
    the 200 held-out validation queries.
    """
    examples = load_examples(VAL_PATH)
    validation_ids = {str(example["query_id"]) for example in examples}

    print(f"Found {len(validation_ids)} validation queries.")

    all_qrels = Qrels.from_file(str(QRELS_PATH), kind="trec")
    filtered_qrels = {query_id: judgments for query_id, judgments in all_qrels.qrels.items() if query_id in validation_ids}

    print(f"Loaded relevance judgments for {len(filtered_qrels)} queries.")

    return Qrels(filtered_qrels)

def evaluate_dense_retrieval():
    """
    Evaluate our fine-tuned Sentence-BERT results
    against the original validation relevance judgments.
    """
    print("Loading validation relevance judgments...")

    qrels = load_validation_qrels()

    print("Loading pretrained Sentence-BERT retrieval results...")

    run = Run.from_file(str(RUN_PATH), kind="trec")
    run.name = "Sentence-BERT (Pretrained)"
    metrics = ["ndcg@10","mrr@10","precision@10"]

    print("\nEvaluating full-collection retrieval...")

    scores = evaluate(qrels, run, metrics, make_comparable=True)

    print("\n--- Pretrained Sentence-BERT Results ---")

    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    return scores

def main():
    print("Starting Sentence-BERT evaluation...")

    evaluate_dense_retrieval()

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()