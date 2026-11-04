"""
Validation data for the IR evaluators used during fine-tuning.

Both models pick their best checkpoint by validation nDCG@10, so they share 
the same queries, corupus, and judgements. Encoding all 2.27M paragraphs at 
every evaluation would take about an hour, so the corupus is a fixed sample: 
every val positive, every val BM25 hard negative, and random distractors 
from the rest of the collection. Scores on this sample are only for 
comparing checkpoints, not for reporting.

Usage:
    from src.training.eval_data import load_ir_eval_data
    queries, corpus, relevant_docs = load_ir_eval_data()
"""
import json
import random
from pathlib import Path

from src.common.paragraphs import load_paragraph_texts
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config
from src.training.data import load_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"
DEFAULT_VAL = PROCESSED_DIR / "training" / "val.json"
DEFAULT_PARAGRAPHS = PROCESSED_DIR / "paragrpaphs.jsonl"
DEFAULT_CLEANING = REPO_ROOT / "config" / "preprocessing_neural.yaml"

def load_all_doc_ids(paragraphs_path):
    with open(paragraphs_path) as f:
        return [json.loads(line)["doc_id"] for line in f]

def load_ir_eval_data(val_path=DEFAULT_VAL, paragraphs_path=DEFAULT_PARAGRAPHS, 
                      num_distractors=20_000, min_relevance=1, with_title=False, seed=42):
    """Return (queries, corpus, relevant_docs) for an InformationRetrievalEvaluator."""
    examples = load_examples(val_path, min_relevance=min_relevance)
    cleaning = load_cleaning_config(DEFAULT_CLEANING)

    # Clean queries the same way search() does at retrieval time.
    queries = {str(ex["query_id"]): clean_for_encoder(ex["query"], **cleaning) 
               for ex in examples}
    relevant_docs = {str(ex["query_id"]): clean_for_encoder(ex["positives"]) 
                     for ex in examples}

    judged = set()
    for ex in examples:
        judged.update(ex["positives"])
        judged.update(ex["negatives"])

    pool = sorted(set(load_all_doc_ids(paragraphs_path)) - judged)
    distractors = random.Random(seed).sample(pool, min(num_distractors, len(pool)))

    corpus = load_paragraph_texts(paragraphs_path, doc_ids=sorted(judged) + distractors, with_title=with_title)
    print(f"Val IR data: {len(queries)} queries, {len(corpus):,} paragraphs "
          f"({len(judged):,} judged + {len(distractors):,} distractors)")
    return queries, corpus, relevant_docs