import argparse
import csv
from pathlib import Path
from datasets import Dataset

from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback

from src.common.paragraphs import load_paragraph_texts
from src.training.data import load_examples
from src.preprocessing.tokenizer import clean_for_encoder, load_cleaning_config
from src.training.eval_data import load_ir_eval_data 


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
    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser(description="Train sparse encoder model.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="run1",
        help="Name of the run. Used to create a unique output directory under checkpoints/sparse/."
    )
    args_cli = parser.parse_args()

    # Define paths relative to repository root
    repo_root = Path(__file__).resolve().parents[2]  
    output_dir = Path("checkpoints/sparse") / args_cli.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = repo_root / "data" / "processed_neural" / "training" / "train.jsonl"
    val_data_path = repo_root / "data" / "processed_neural" / "training" / "val.jsonl"
    paragraphs_path = repo_root / "data" / "processed_neural" / "paragraphs.jsonl"

    cleaning = load_cleaning_config(repo_root / "config" / "preprocessing_neural.yaml")
    train_dataset = build_triples(train_data_path, paragraphs_path, cleaning)   # ~5,138 rows

    # Updated: Load validation data using teammate's fixed eval_data module with 20k distractors
    queries, corpus, relevant_docs = load_ir_eval_data()

    # Updated: Added show_progress_bar=True so evaluation output is visible
    val_evaluator = SparseInformationRetrievalEvaluator(
        queries=queries, 
        corpus=corpus, 
        relevant_docs=relevant_docs, 
        name="val",
        show_progress_bar=True
    )

    # Training Arguments with Early Stopping and evaluation settings
    args = SparseEncoderTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_dot_ndcg@10",
        greater_is_better=True,
        logging_steps=10,
        per_device_train_batch_size=32,
        fp16=True,
        learning_rate=1e-5,
        num_train_epochs=3,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    model = SparseEncoder(model_name_or_path="naver/splade-cocondenser-ensembledistil")
    
    base_loss = SparseMultipleNegativesRankingLoss(model=model)
    
    # Pass model as the first argument, followed by the base loss and regularizer weight
    loss = SpladeLoss(
        model,
        loss=base_loss,
        document_regularizer_weight=0.001
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,  # We are passing val_evaluator directly, no need for val_dataset here
        loss=loss,
        evaluator=val_evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"Output directory for this run: {output_dir}")
    print("Running zero-shot evaluation baseline...")
    zero_shot_metrics = trainer.evaluate()
    print(f"Zero-shot baseline metrics: {zero_shot_metrics}")

    print("Starting fine-tuning...")
    trainer.train()

    # Save the best fine-tuned model checkpoint explicitly
    best_model_path = output_dir / "best_model"
    model.save(str(best_model_path))
    print(f"Best fine-tuned model saved to {best_model_path}")

    # Write trainer.state.log_history to a CSV file for tracking and plotting
    log_history_path = output_dir / "training_log_history.csv"
    print(f"Saving training log history to {log_history_path}")
    
    with open(log_history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "split", "loss"])
        for entry in trainer.state.log_history:
            if "loss" in entry:
                writer.writerow([entry["step"], entry["epoch"], "train", entry["loss"]])
            elif "eval_loss" in entry:
                writer.writerow([entry["step"], entry["epoch"], "val", entry["eval_loss"]])


if __name__ == "__main__":
    main()