"""
Fine-tune SPLADE on the Train queries.

Starts from naver/splade-cocondenser-selfdistil (the same model as the
zero-shot index) and trains on (query, positive, BM25 hard negative) triples
built from data/processed_neural/training/:

    train.jsonl  training triples, one row per judged positive
    val.jsonl    validation triples (validation loss, for the loss plot), and
                 the shared validation IR data from src/training/eval_data.py
                 (val nDCG@10, for checkpointing and early stopping)

The loss is SpladeLoss: SparseMultipleNegativesRankingLoss (each query's
positive against its hard negative and the other paragraphs in the batch),
plus FLOPS regularization to keep query and document vectors sparse.

Writes to checkpoints/sparse/<run-name>/:
    best_model/                 the checkpoint with the best val nDCG@10
    training_log_history.csv    step, epoch, split, loss (train and val)
    summary.json                zero-shot and best val metrics, settings

Usage, from the repository root:
    python -m src.training.train_splade --run-name run2
    python -m src.training.train_splade --run-name bs16 --batch-size 16
"""
import argparse
import csv
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback

from src.common.paragraphs import load_paragraph_texts
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config
from src.training.data import load_examples
from src.training.eval_data import load_ir_eval_data

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"
TRAINING_DIR = PROCESSED_DIR / "training"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "sparse"
CLEANING_CONFIG = REPO_ROOT / "config" / "preprocessing_neural.yaml"

DEFAULT_MODEL = "naver/splade-cocondenser-selfdistil"
MAX_SEQ_LENGTH = 256           # same as the paragraph index
BEST_METRIC = "eval_val_dot_ndcg@10"


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
            neg_id = ex["negatives"][i % len(ex["negatives"])]  # cycle through the hard negatives
            rows["anchor"].append(query)
            rows["positive"].append(texts[pos_id])
            rows["negative"].append(texts[neg_id])
    print(f"Built {len(rows['anchor']):,} triples from {len(examples):,} queries in {data_path.name}")
    return Dataset.from_dict(rows)


def write_loss_csv(log_history, path):
    """Write train and validation loss as step, epoch, split, loss (for plot_loss.py)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "split", "loss"])
        for entry in log_history:
            if "loss" in entry:
                writer.writerow([entry["step"], entry.get("epoch", 0), "train", entry["loss"]])
            elif "eval_loss" in entry:
                writer.writerow([entry["step"], entry.get("epoch", 0), "val", entry["eval_loss"]])


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SPLADE on the Train queries.")
    parser.add_argument("--run-name", default="run1", help="Output folder under checkpoints/sparse/ (default: run1)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=32, help="Triples per batch; more means more in-batch negatives (default: 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum epochs; early stopping may end sooner (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate and checkpoint every N steps (default: 50)")
    parser.add_argument("--patience", type=int, default=3, help="Evaluations without improvement before stopping (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = CHECKPOINT_DIR / args.run_name
    if (output_dir / "best_model").exists():
        raise SystemExit(f"{output_dir} already has a best_model; pick another --run-name.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data: triples for the losses, shared val IR data for nDCG@10.
    paragraphs_path = PROCESSED_DIR / "paragraphs.jsonl"
    cleaning = load_cleaning_config(CLEANING_CONFIG)
    train_dataset = build_triples(TRAINING_DIR / "train.jsonl", paragraphs_path, cleaning)  # ~5,138 rows
    val_dataset = build_triples(TRAINING_DIR / "val.jsonl", paragraphs_path, cleaning)      # ~582 rows

    queries, corpus, relevant_docs = load_ir_eval_data()
    evaluator = SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val",
        show_progress_bar=True,
    )

    # Model and loss.
    model = SparseEncoder(args.model)
    model.max_seq_length = MAX_SEQ_LENGTH
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=5e-5,
        document_regularizer_weight=3e-5,
    )

    training_args = SparseEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # no query twice in a batch, so its other positives aren't used as negatives
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,                  # must be a multiple of eval_steps for load_best_model_at_end
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=10,
        seed=42,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   # validation loss (required for the loss plot)
        loss=loss,
        evaluator=evaluator,        # val nDCG@10 (checkpointing and early stopping)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # Zero-shot baseline. Also gives the step-0 point of the validation loss curve.
    print("Evaluating the base model before fine-tuning...")
    zero_shot = trainer.evaluate()
    print(f"Zero-shot {BEST_METRIC}: {zero_shot[BEST_METRIC]:.4f}, val loss: {zero_shot['eval_loss']:.4f}")

    print("Fine-tuning...")
    trainer.train()

    # load_best_model_at_end restored the best checkpoint into trainer.model.
    best_model_path = output_dir / "best_model"
    trainer.model.save(str(best_model_path))
    print(f"Best model ({BEST_METRIC} = {trainer.state.best_metric:.4f}) saved to {best_model_path}")

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
            "max_seq_length": MAX_SEQ_LENGTH,
            "query_regularizer_weight": 5e-5,
            "document_regularizer_weight": 3e-5,
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