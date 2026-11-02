"""
This file prepares our training data and fine-tunes Sentence-BERT.

We use the training queries and relevance judgments to teach the model
which papers and paragraphs are relevant to each query.
"""

import json
from pathlib import Path
import random
from datasets import Dataset
from sentence_transformers import(SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator

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
    paragraph_documents = load_document_texts("data/processed/paragraphs.jsonl", paragraph_ids, "paragraph")

    return paper_documents, paragraph_documents

def create_training_arguments():
    """
    Set up the training settings for Sentence-BERT.
    """
    return SentenceTransformerTrainingArguments(
        output_dir="models/finetuned_dense_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=42,
        report_to="none"
    )

def load_training_model():
    """
    Load the pretrained Sentence-BERT model for fine-tuning.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_training_loss(model):
    """
    Set up the loss function for fine-tuning Sentence-BERT.
    """
    return losses.MultipleNegativesRankingLoss(model)

def create_training_trainer(model, dataset, evaluator=None):
    """
    Set up the Sentence-BERT trainer using our training dataset.
    """
    arguments = create_training_arguments()
    loss = create_training_loss(model)

    return SentenceTransformerTrainer(model=model, args=arguments, train_dataset=dataset, loss=loss, evaluator=evaluator)

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
    paragraph_documents = load_document_texts("data/processed/paragraphs.jsonl", paragraph_ids, "paragraph")

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