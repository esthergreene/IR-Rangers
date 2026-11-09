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
from src.common.trec import run_filename
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config
from src.training.data import load_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"
DEFAULT_VAL = PROCESSED_DIR / "training" / "val.jsonl"
DEFAULT_PARAGRAPHS = PROCESSED_DIR / "paragraphs.jsonl"
DEFAULT_CLEANING = REPO_ROOT / "config" / "preprocessing_neural.yaml"
DEFAULT_CANDIDATE_RUNS = (REPO_ROOT / "runs" / run_filename("paragraph", "bm25", "train"),)

def load_all_doc_ids(paragraphs_path):
    with open(paragraphs_path) as f:
        return [json.loads(line)["doc_id"] for line in f]

def load_run_candidates(run_paths, query_ids, depth):
    """
    Return the doc_ids in the top `depth` of each run, for the given queries.

    Run files are written in rank order, so the first `depth` lines per query 
    are its top `depth`.
    """
    candidates = set()
    for run_path in run_paths:
        seen = {}
        with open(run_path, "r", encoding="utf-8") as f:
            for line in f:
                query_id, _, doc_id, *_ = line.split()
                if query_id in query_ids and seen.get(query_id, 0) < depth:
                    seen[query_id] = seen.get(query_id, 0) + 1
                    candidates.add(doc_id)
        missing = query_ids - set(seen)
        if missing:
            print(f"Warning: {len(missing)} val queries have no lines in {run_path.name}")
    return candidates

def load_ir_eval_data(val_path=DEFAULT_VAL, paragraphs_path=DEFAULT_PARAGRAPHS, candidate_runs=DEFAULT_CANDIDATE_RUNS,
                      candidate_depth=100, num_distractors=10_000, min_relevance=1, with_title=False, seed=42):
    """
    Return (queries, corpus, relevant_docs) for an InformationRetrievalEvaluator.
    
    The corpus is every judged paragraph, each val query's top `candidate_depth` 
    from `candidate_runs` (the near-misses a model must beat at search time), 
    and `num_distractors` random paragraphs as background. Pass candidate_depth=0, 
    num_distractors=20_000 to rebuild the run2 corpus.
    """
    examples = load_examples(val_path, min_relevance=min_relevance)
    cleaning = load_cleaning_config(DEFAULT_CLEANING)

    # Clean queries the same way search() does at retrieval time.
    queries = {str(ex["query_id"]): clean_for_encoder(ex["query"], **cleaning) 
               for ex in examples}
    relevant_docs = {str(ex["query_id"]): set(ex["positives"]) for ex in examples}

    judged = set()
    for ex in examples:
        judged.update(ex["positives"])
        judged.update(ex["negatives"])

    candidates = set()
    if candidate_depth:
        candidates = load_run_candidates(candidate_runs, set(queries), candidate_depth) - judged

    pool = sorted(set(load_all_doc_ids(paragraphs_path)) - judged - candidates)
    distractors = random.Random(seed).sample(pool, min(num_distractors, len(pool)))

    corpus = load_paragraph_texts(paragraphs_path, doc_ids=sorted(judged) + sorted(candidates) + distractors, with_title=with_title)
    print(f"Val IR data: {len(queries)} queries, {len(corpus):,} paragraphs "
          f"({len(judged):,} judged + {len(candidates)} retrieved + {len(distractors):,} random)")
    return queries, corpus, relevant_docs