"""
This file builds FAISS indexes for our paper and paragraph collections.

It reads the documents, converts their text into Sentence-BERT embeddings,
and stores those embeddings so we can search them later.
"""

import json
from pathlib import Path
import faiss
import numpy as np
from src.model.dense import DenseModel
from src.common.paragraphs import iter_paragraph_batches

def load_documents(file_path):
    """
    This reads documents from one of our JSONL collections.

    Args:
        file_path: The path to our paper or paragraph collection.

    Yields:
        One document at a time.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

def get_document_text(document, collection):
    """
    This gets the text we want to encode for each document.

    Args:
        document: A paper or paragraph from our collection.
        collection: Either paper or paragraph.

    Returns:
        The document text.
    """
    if collection == "paper":
        title = document.get("title", "")
        abstract = document.get("abstract", "")

        return f"{title} {abstract}".strip()
    elif collection == "paragraph":
        return document.get("text", "")
    else:
        raise ValueError("Collection must be 'paper' or 'paragraph'.")

def document_batches(file_path, collection, batch_size=32):
    """
    This groups our documents into smaller batches for Sentence-BERT.

    Args:
        file_path: The path to our document collection.
        collection: Either paper or paragraph.
        batch_size: How many documents to process at once.

    Yields:
        A batch of document IDs and their text.
    """
    doc_ids = []
    texts = []

    for document in load_documents(file_path):
        text = get_document_text(document, collection)

        if not text.strip():
            continue

        doc_ids.append(document["doc_id"])
        texts.append(text)

        if len(texts) == batch_size:
            yield doc_ids, texts
            doc_ids = []
            texts = []

    if texts:
        yield doc_ids, texts

def build_dense_index(file_path, collection, output_dir, model_path="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, checkpoint_interval=50000):
    """
    This builds a FAISS index for our paper or paragraph collection.

    Args:
        file_path: The path to our JSONL collection.
        collection: Either paper or paragraph.
        output_dir: Where we want to save the index.
        model_path: The Sentence-BERT model we're using.
        batch_size: How many documents to encode at once.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = DenseModel(model_path)
    dimension = model.model.get_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    doc_ids = []
    checkpoint_path = output_dir / f"{collection}_dense_checkpoint.faiss"
    checkpoint_ids_path = output_dir / f"{collection}_dense_checkpoint_ids.json"

    if checkpoint_path.exists() and checkpoint_ids_path.exists():
        index = faiss.read_index(str(checkpoint_path))

        with open(checkpoint_ids_path, "r", encoding="utf-8") as file:
            doc_ids = json.load(file)

        if len(doc_ids) < index.ntotal:
            raise ValueError("Checkpoint is missing document IDs.")

        if len(doc_ids) > index.ntotal:
            doc_ids = doc_ids[:index.ntotal]
            print("Recovered document IDs from an interrupted checkpoint.")

        print(f"Loaded checkpoint with {len(doc_ids)} documents")

    if collection == "paragraph":
        batches = iter_paragraph_batches(file_path, batch_size=batch_size, with_title=False)
    else:
        batches = document_batches(file_path, collection, batch_size)

    processed_count = 0

    for batch_ids, batch_texts in batches:
        batch_end = processed_count + len(batch_ids)

        if batch_end <= len(doc_ids):
            processed_count = batch_end
            continue

        if processed_count < len(doc_ids):
            skip_count = len(doc_ids) - processed_count
            batch_ids = batch_ids[skip_count:]
            batch_texts = batch_texts[skip_count:]

        processed_count = batch_end

        embeddings = model.encode_documents(batch_texts, batch_size=batch_size)
        index.add(embeddings)
        doc_ids.extend(batch_ids)

        if len(doc_ids) % 10000 < batch_size:
            print(f"Indexed {len(doc_ids)} documents")

        if len(doc_ids) // checkpoint_interval > (len(doc_ids) - len(batch_ids)) // checkpoint_interval:
            temporary_index_path = checkpoint_path.with_suffix(".tmp")
            faiss.write_index(index, str(temporary_index_path))
            temporary_ids_path = checkpoint_ids_path.with_suffix(".tmp")

            with open(temporary_ids_path, "w", encoding="utf-8") as file:
                json.dump(doc_ids, file)

            temporary_ids_path.replace(checkpoint_ids_path)
            temporary_index_path.replace(checkpoint_path)

            print(f"Checkpoint saved at {len(doc_ids)} documents")

    index_path = output_dir / f"{collection}_dense.faiss"
    ids_path = output_dir / f"{collection}_dense_ids.json"
    faiss.write_index(index, str(index_path))

    with open(ids_path, "w", encoding="utf-8") as file:
        json.dump(doc_ids, file)

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved {len(doc_ids)} documents IDS to {ids_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a FAISS index over the paragraph collection.")
    parser.add_argument("--paragraphs", default="data/processed_neural/paragraphs.jsonl")
    parser.add_argument("--output-dir", required=True, help="One folder per model, e.g., data/processed_neural/dense_run1")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    build_dense_index(args.paragraphs, "paragraph", args.output_dir, args.model, args.batch_size)