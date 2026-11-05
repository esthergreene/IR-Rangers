import csv
from pathlib import Path
import torch
from datasets import Dataset

from sentence_transformers.trainer import SentenceTransformerTrainer as SparseEncoderTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments as SparseEncoderTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss as SparseMultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator as SparseInformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback

from src.common.paragraphs import load_paragraph_texts
from src.training.data import load_examples
from src.index.sparse_indexer import SpladeEncoder


def build_dataset_and_evaluator(data_path, paragraphs_path):
    examples = load_examples(data_path)
    
    # 1. Collect ONLY document IDs for the corpus text mapping
    doc_ids = set()
    for ex in examples:
        if isinstance(ex.get("positives"), dict):
            doc_ids.update(ex["positives"].keys())
        if isinstance(ex.get("negatives"), list):
            doc_ids.update(ex["negatives"])
            
    print(f"Loading texts for {len(doc_ids):,} document corpus IDs from {paragraphs_path}...")
    text_mapping = load_paragraph_texts(paragraphs_path, doc_ids=doc_ids)

    # 2. Prepare Training Dataset (anchor = query text string from record)
    anchors, positives, negatives = [], [], []
    for ex in examples:
        q_text = ex.get("query")  # Query text string is right in the example record
        
        pos_dict = ex.get("positives", {})
        pos_id = list(pos_dict.keys())[0] if pos_dict else None
        pos_text = text_mapping.get(pos_id) if pos_id else None
        
        neg_list = ex.get("negatives", [])
        neg_id = neg_list[0] if neg_list else None
        neg_text = text_mapping.get(neg_id) if neg_id else None
        
        if q_text and pos_text and neg_text:
            anchors.append(q_text)
            positives.append(pos_text)
            negatives.append(neg_text)

    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })

    # 3. Prepare Evaluator Data Structures
    queries = {ex["query_id"]: ex["query"] for ex in examples}
    corpus = {doc_id: text_mapping[doc_id] for doc_id in doc_ids if doc_id in text_mapping}
    
    relevant_docs = {}
    for ex in examples:
        q_id = ex["query_id"]
        pos_ids = ex.get("positives", {}).keys() if isinstance(ex.get("positives"), dict) else []
        for pos_id in pos_ids:
            if pos_id in corpus:
                if q_id not in relevant_docs:
                    relevant_docs[q_id] = set()
                relevant_docs[q_id].add(pos_id)

    evaluator = SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val"
    )

    return train_dataset, evaluator


def main():
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed_neural"
    paragraphs_path = processed_dir / "paragraphs.jsonl"
    
    train_data_path = processed_dir / "training" / "train.jsonl"
    val_data_path = processed_dir / "training" / "val.jsonl"
    
    output_dir = repo_root / "checkpoints" / "sparse" / "run1"

    print("Preparing training dataset and evaluator...")
    train_dataset, _ = build_dataset_and_evaluator(train_data_path, paragraphs_path)
    _, val_evaluator = build_dataset_and_evaluator(val_data_path, paragraphs_path)

    # Initialize model using your project's local SpladeEncoder wrapper
    model_name = "naver/splade-cocondenser-selfdistil"
    print(f"Loading local SpladeEncoder model: {model_name}")
    
    model = SpladeEncoder(model_name)
    
    # 1. Attach a tokenize method expected by SentenceTransformerTrainer
    if not hasattr(model, "tokenize"):
        def _tokenize(texts, **kwargs):
            return model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                **kwargs
            )
        model.tokenize = _tokenize

    # 2. Attach a .to() method expected by Hugging Face Trainer
    if not hasattr(model, "to"):
        def _to(device):
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to(device)
            return model
        model.to = _to

    # 3. Attach a forward() method expected by Hugging Face Trainer
    if not hasattr(model, "forward") and hasattr(model, "model"):
        def _forward(*args, **kwargs):
            return model.model(*args, **kwargs)
        model.forward = _forward

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set up loss function: MultipleNegativesRankingLoss
    loss = SparseMultipleNegativesRankingLoss(model=model)

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
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        num_train_epochs=3,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_evaluator,
        loss=loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

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
    
    if trainer.state.log_history:
        keys = set().union(*(d.keys() for d in trainer.state.log_history))
        with open(log_history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            for entry in trainer.state.log_history:
                writer.writerow(entry)


if __name__ == "__main__":
    main()