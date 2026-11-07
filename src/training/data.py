"""
Loading the training and validation examples built by 
scripts/build_training_data.py.

Queries are stored as written in Train.json. Clean them with 
clean_for_encoder() before encoding, the same way queries are cleaned at 
search time.

Usage:
    from src.training.data import load_examples
    train = load_examples("data/processed_neural/training/train.jsonl")
    val = load_examples("data/processed_neural/training/val.jsonl")
"""
import json

def load_examples(path, min_relevance=1):
    """
    Load training examples, keeping positives at or above a relevance grade.

    Args:
        path (str or Path): train.jsonl or val.jsonl.
        min_relevance (int): 1 keeps partially and fully relevant positives; 
            2 keeps only fully relevant ones.

    Returns:
        list of dict: Examples with "query_id", "query", "label", "positives"
            (doc_id -> grade), and "negatives" (doc_ids). Examples left with 
            no positive at this grade are dropped.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            example["positives"] = {doc_id: grade for doc_id, grade in example["positives"].items() 
                                    if grade >= min_relevance}
            if example["positives"]:
                examples.append(example)
    return examples