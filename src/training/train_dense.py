"""
Fine-tune Sentence-BERT (dense retrieval) on the Train queries.

We use the training queries and relevance judgments to teach the model
which papers are relevant to each query.

Trains on (query, positive, BM25 hard negative) triples from 
data/processed_neural/training/:

    train.jsonl     training triples, one row per judged positive
    val.jsonl       validation triples (validation loss, for the loss plot), and 
                    the shared validation IR data from src/training/eval_data.py 
                    (val nDCG@10, for checkpointing and early stopping)

Writes to checkpoints/dense/<run-name>/:
    best_model/                 the checkpoint with the best val nDCG@10
    training_log_history.csv    step, epoch, split, loss (train and val)
    summary.json                zero-shot and best val metrics, settings

Usage, from the repository root:
    python -m src.training.train_dense --run-name run1
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import EarlyStoppingCallback

from src.training.data import load_examples
from src.training.eval_data import load_ir_eval_data
from src.common.paragraphs import load_paragraph_texts
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"
TRAINING_DIR = PROCESSED_DIR / "training"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "dense"
CLEANING_CONFIG = REPO_ROOT / "config" / "preprocessing_neural.yaml"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BEST_METRIC = "eval_val_cosine_ndcg@10"

def build_triples(data_path, paragraphs_path, cleaning):
    """One (query, positive, hard negative) row per judged positive."""
    examples = load_examples(data_path)
    doc_ids = set()
    for ex in examples:
        doc_ids.update(ex["positives"])
        doc_ids.update(ex["negatives"])
    texts = load_paragraph_texts(paragraphs_path, doc_ids=doc_ids)

    rows = {"anchor": [], "positive": [], "negative": []}
    for ex in examples:
        query = clean_for_encoder(ex["query"], **cleaning)
        for i, pos_id in enumerate(ex["positives"]):
            neg_id = ex["negatives"][i % len(ex["negatives"])] # cycle through the hard negatives
            rows["anchor"].append(query)
            rows["positive"].append(texts[pos_id])
            rows["negative"].append(texts[neg_id])
    print(f"Built {len(rows['anchor']):,} triples from {len(examples):,} queries in {data_path.name}")
    return Dataset.from_dict(rows)

def write_loss_csv(log_history, path):
    """Write train and validation loss as step, epoch, split, loss."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "split", "loss"])
        for entry in log_history:
            if "loss" in entry:
                writer.writerow([entry["step"], entry.get("epoch", 0), "train", entry["loss"]])
            elif "eval_loss" in entry:
                writer.writerow([entry["step"], entry.get("epoch", 0), "val", entry["eval_loss"]])

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Sentence-BERT on the Train queries.")
    parser.add_argument("--run-name", default="run1", help="Output folder under checkpoints/dense/ (default: run1)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=32, help="Triples per batch, more means more in-batch negatives (default: 32)")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum epochs; early stopping may end sooner (default: 5)")
    parser.add_argument("--learning-rate", type=float, defualt=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate and checkpoint every N steps (defualt: 50)")
    parser.add_argument("--patience", type=int, default=3, help="Evaluations without improvement before stopping (default: 3)")
    return parser.parse_args()

def load_training_queries(file_path):
    """
    Load our training queries and organize them by query ID.

    Args:
        file_path: Path to Train.json.

    Returns:
        A dictionary matching each query ID to its question.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        queries = json.load(file)

    return {
        str(query["query_id"]): query["query"]
        for query in queries
    }

def load_training_qrels(file_path):
    """
    Load the relevance judgments for our training queries.

    Args:
        file_path: Path to a training qrels file.

    Returns:
        A list of query IDs, document IDs, and relevance scores.
    """
    judgements = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            query_id, _, doc_id, relevance = line.split()
            judgements.append((query_id, doc_id, int(relevance)))

    return judgements

def load_document_texts(file_path, relevant_ids, collection):
    """
    Load the text for documents referenced in our training qrels.

    Args:
        file_path: Path to the paper or paragraph JSONL collection.
        relevant_ids: Document IDs we need for training.
        collection: Either paper or paragraph.

    Returns:
        A dictionary matching document IDs to their text.
    """
    documents = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)
            doc_id = document ["doc_id"]

            if doc_id not in relevant_ids:
                continue

            if collection == "paper":
                text = f"{document.get('title', '')} {document.get('abstract', '')}".strip()
            elif collection == "paragraph":
                text = document.get("text", "").strip()
            else:
                raise ValueError("Collection must be 'paper' or 'paragraph'.")

            if text:
                documents[doc_id] = text

    return documents

def create_training_pairs(queries, judgements, documents):
    """
    Match training queries with their relevant document text.

    Returns:
        A list of query and document text pairs.
    """
    pairs = []

    for query_id, doc_id, relevance in judgements:
        if relevance <= 0:
            continue

        if query_id not in queries or doc_id not in documents:
            continue

        query_text = queries[query_id]
        document_text = documents[doc_id]
        pairs.append((query_text, document_text))

    return pairs

def combine_training_pairs(paper_pairs, paragraph_pairs):
    """
    Combine paper and paragraph training pairs into one dataset.
    """
    return paper_pairs + paragraph_pairs

def split_training_queries(queries, validation_ratio=0.2, seed=42):
    """
    Split query IDs into training and validation groups.

    Using query IDs prevents the same question from appearing
    in both groups.
    """
    query_ids = list(queries.keys())
    random.Random(seed).shuffle(query_ids)
    split_index = int(len(query_ids) * (1 - validation_ratio))
    train_ids = set(query_ids[:split_index])
    validation_ids = set(query_ids[split_index:])

    return train_ids, validation_ids

def split_training_qrels(judgements, train_ids, validation_ids):
    """
    Separate relevance judgments using our training and validation query IDs.
    """
    train_qrels = []
    validation_qrels = []

    for query_id, doc_id, relevance in judgements:
        if query_id in train_ids:
            train_qrels.append((query_id, doc_id, relevance))
        elif query_id in validation_ids:
            validation_qrels.append((query_id, doc_id, relevance))

    return train_qrels, validation_qrels

def create_training_dataset(pairs):
    """
    Convert query-document pairs into a Hugging Face dataset.
    """
    return Dataset.from_dict({
        "anchor": [query for query, document in pairs],
        "positive": [document for query, document in pairs]
    })

def prepare_training_dataset(queries, paper_qrels, paragraph_qrels, paper_documents, paragraph_documents):
    """
    Prepare one training dataset using both papers and paragraphs.
    """
    paper_pairs = create_training_pairs(queries, paper_qrels, paper_documents)
    paragraph_pairs = create_training_pairs(queries, paragraph_qrels, paragraph_documents)
    all_pairs = combine_training_pairs(paper_pairs, paragraph_pairs)

    return create_training_dataset(all_pairs)

def prepare_paragraph_training_dataset(queries, paragraph_qrels, paragraph_documents):
    """
    Prepare training examples using only relevant paragraphs.
    """
    paragraph_pairs = create_training_pairs(queries, paragraph_qrels, paragraph_documents)

    return create_training_dataset(paragraph_pairs)

def load_and_split_training_data():
    """
    Load our training queries and relevance judgments,
    then split them into training and validation groups.
    """
    queries = load_training_queries("data/raw/Train.json")

    paper_qrels = load_training_qrels("/home/esther.greene/datasets/IR26/Train_paper_qrel.tsv")
    paragraph_qrels = load_training_qrels("/home/esther.greene/datasets/IR26/Train_paragraph_qrel.tsv")
    train_ids, validation_ids = split_training_queries(queries)
    train_paper_qrels, validation_paper_qrels = split_training_qrels(paper_qrels, train_ids, validation_ids)
    train_paragraph_qrels, validation_paragraph_qrels = split_training_qrels(paragraph_qrels, train_ids, validation_ids)

    return {
        "queries": queries,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "train_paper_qrels": train_paper_qrels,
        "validation_paper_qrels": validation_paper_qrels,
        "train_paragraph_qrels": train_paragraph_qrels,
        "validation_paragraph_qrels": validation_paragraph_qrels
    }

def load_training_documents(data):
    """
    Load the paper and paragraph texts needed for training.
    """
    paper_ids = {doc_id for _, doc_id, _ in data["train_paper_qrels"]}
    paragraph_ids = {doc_id for _, doc_id, _ in data["train_paragraph_qrels"]}
    paper_documents = load_document_texts("data/processed/papers.jsonl", paper_ids, "paper")
    paragraph_documents = load_document_texts("data/processed_neural/paragraphs.jsonl", paragraph_ids, "paragraph")

    return paper_documents, paragraph_documents

def create_training_arguments(args, output_dir, num_train_examples):
    """
    Set up the training settings for Sentence-BERT.
    """
    total_steps = math.ceil(num_train_examples / args.batch_size) * args.epochs
    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=(0.1 * total_steps),
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES, # no query twice in a batch, so its other postives aren't used as negatives
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,   # must be a multiple of eval_steps for load_best_model_at_end
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=10,
        seed=42,
        report_to="none"
    )

def load_training_model(model_name):
    """
    Load the pretrained Sentence-BERT model for fine-tuning.
    """
    return SentenceTransformer(model_name)

def create_training_loss(model):
    """
    MultipleNegativesRankingLoss: each query's positive against its hard 
    negative and every other paragraph in the batch.
    """
    return losses.MultipleNegativesRankingLoss(model)

def create_training_trainer(model, training_args, train_dataset, val_dataset, evaluator, patience):
    """
    Set up the Sentence-BERT trainer with validation loss, nDCG@10 and early stopping.
    """
    return SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   # validation loss (required for loss plot)
        loss=create_training_loss(model),
        evaluator=evaluator,        # val nDCG@10 (checkpointinng and early stopping)
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience)
        ]
    )

def prepare_validation_dataset(data, paper_documents, paragraph_documents):
    """
    Prepare validation examples without including them in training.
    """
    return prepare_training_dataset(
        data["queries"],
        data["validation_paper_qrels"],
        data["validation_paragraph_qrels"],
        paper_documents,
        paragraph_documents
    )

def load_validation_documents(data):
    """
    Load the paper and paragraph texts needed for validation.
    """
    paper_ids = {doc_id for _, doc_id, _ in data["validation_paper_qrels"]}
    paragraph_ids = {doc_id for _, doc_id, _ in data["validation_paragraph_qrels"]}
    paper_documents = load_document_texts("data/processed/papers.jsonl", paper_ids, "paper")
    paragraph_documents = load_document_texts("data/processed_neural/paragraphs.jsonl", paragraph_ids, "paragraph")

    return paper_documents, paragraph_documents

def get_validation_queries(data):
    """
    Get the held-out queries for retrieval evaluation.
    """
    return {query_id: data["queries"][query_id] for query_id in data["validation_ids"]}

def get_validation_corpus(paper_documents, paragraph_documents):
    """
    Combine validation papers and paragraphs into one retrieval corpus.
    """
    corpus = {}

    for doc_id, text in paper_documents.items():
        corpus[f"paper_{doc_id}"] = text

    for doc_id, text in paragraph_documents.items():
        corpus[f"paragraph_{doc_id}"] = text

    return corpus

def get_paragraph_validation_corpus(paragraph_documents):
    """
    Build a validation corpus containing only paragraphs.
    """
    return dict(paragraph_documents)

def load_background_paragraphs(file_path):
    """
    Load background paragraphs for validation.
    """
    paragraphs = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)
            paragraphs[document["doc_id"]] = document["text"]

    return paragraphs

def build_paragraph_validation_corpus(validation_paragraphs, background_paragraphs):
    """
    Combine validation paragraphs with background paragraphs.
    """
    corpus = dict(background_paragraphs)
    corpus.update(validation_paragraphs)

    return corpus

def get_validation_relevant_docs(data, corpus):
    """
    Match validation queries with their relevant documents.
    """
    relevant_docs = {query_id: set() for query_id in data["validation_ids"]}

    for query_id, doc_id, relevance in data["validation_paper_qrels"]:
        document_id = f"paper_{doc_id}"

        if relevance > 0 and document_id in corpus:
            relevant_docs[query_id].add(document_id)

    for query_id, doc_id, relevance in data["validation_paragraph_qrels"]:
        document_id = f"paragraph_{doc_id}"

        if relevance > 0 and document_id in corpus:
            relevant_docs[query_id].add(document_id)

    return relevant_docs

def get_paragraph_validation_relevant_docs(data, corpus):
    """
    Match validation queries with their relevant paragraphs.
    """
    relevant_docs = {query_id: set() for query_id in data["validation_ids"]}

    for query_id, doc_id, relevance in data["validation_paragraph_qrels"]:
        if relevance > 0 and doc_id in corpus:
            relevant_docs[query_id].add(doc_id)

    return relevant_docs

def create_validation_evaluator(queries, corpus, relevant_docs):
    """
    Set up retrieval evaluation using our held-out queries.
    """
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="validation",
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[5,10]
    )

def load_shared_training_dataset():
    """
    Prepare Sentence-BERT training pairs using our
    shared training split and cleaned paragraph collection.
    """
    examples = load_examples("data/processed_neural/training/train.jsonl")
    cleaning = load_cleaning_config("config/preprocessing_neural.yaml")
    positive_ids = {doc_id for example in examples for doc_id in example["positives"]}
    paragraphs = load_paragraph_texts("data/processed_neural/paragraphs.jsonl", doc_ids=positive_ids, with_title=False)
    pairs = []

    for example in examples:
        query = clean_for_encoder(example["query"], **cleaning)

        for doc_id in example["positives"]:
            [pairs.append((query, paragraphs[doc_id]))]

    dataset = create_training_dataset(pairs)

    print(f"Loaded {len(examples)} training queries.")
    print(f"Created {len(dataset)} Sentence-BERT training pairs.")

    return dataset

def create_shared_validation_evaluator():
    """
    Create the Sentence-BERT evaluator using our
    shared validation queries and paragraph corpus.

    (val nDCG@10, for checkpointing and early stopping)
    """
    queries, corpus, relevant_docs = load_ir_eval_data()
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="validation",
        batch_size=128,
        show_progress_bar=True,
    )

def main():
    """
    Fine-tune Sentence-BERT using our shared training and validation datasets.
    """
    args = parse_args()
    output_dir = CHECKPOINT_DIR / args.run_name
    if (output_dir / "best_model").exists():
        raise SystemExit(f"{output_dir} already has a best model; pick another --run-name.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data: triples for the losses, shared val IR data for nDCG@10
    paragraphs_path = PROCESSED_DIR / "paragraphs.jsonl"
    cleaning = load_cleaning_config(CLEANING_CONFIG)
    train_dataset = build_triples(TRAINING_DIR / "train.jsonl", paragraphs_path, cleaning)
    val_dataset = build_triples(TRAINING_DIR / "val.jsonl", paragraphs_path, cleaning)
    evaluator = create_shared_validation_evaluator()

    model = load_training_model(args.model)
    training_args = create_training_arguments(args, output_dir, len(train_dataset))
    trainer = create_training_trainer(model, training_args, train_dataset, val_dataset, 
                                      evaluator, args.patience)

    print("Evaluating the base model before fine-tuning...")
    zero_shot = trainer.evaluate()
    print(f"Zero-shot {BEST_METRIC}: {zero_shot[BEST_METRIC]:.4f}, val loss: {zero_shot['eval_loss']:.4f}")

    print("Starting Sentence-BERT fine-tuning...")
    trainer.train()

    # load_best_model_at_end restored the best checkpoint into trainer.model.
    best_model_path = output_dir / "best_model"
    trainer.model.save(str(best_model_path))

    log_path = output_dir / "training_log_history.csv"
    write_loss_csv(trainer.state.log_history, log_path)
    print(f"Loss history saved to {log_path}")

    final = trainer.evaluate()
    summary = {
        "base_model": args.model,
        "settings": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "max_seq_length": model.max_seq_length,
        },
        "train_triples": len(train_dataset),
        "val_triples": len(val_dataset),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "zero_shot": {key: value for key, value in zero_shot.items() if key.startswith("eval_")},
        "best": {key: value for key, value in final.items() if key.startswith("eval_")},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Zero-shot -> best {BEST_METRIC}: {zero_shot[BEST_METRIC]:.4f} -> {final[BEST_METRIC]:.4f}")
    print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()