"""
Building and searching the SPLADE sparse paragraph index (Part 2).

Each paragraph in data/processed_neural/paragraphs.jsonl is encoded by a
SPLADE model into a sparse vector over the model's vocabulary (30,522
WordPiece terms), and the vectors are stored as one scipy sparse matrix:
one row per paragraph, one column per vocabulary term. Row i is the i-th
paragraph in the file.

The matrix is stored column by column (CSC), so a search only reads the
columns of the query's terms, which works like term-at-a-time scoring over
an inverted index. Scores are dot products, as SPLADE is trained.

Saved files:
    sparse_paragraph_index.npz          the matrix
    sparse_paragraph_index.meta.json    paragraph IDs (row order) and the
                                        settings queries must be encoded with

Usage, from the repository root:
    python -m src.index.sparse_indexer
    python -m src.index.sparse_indexer --limit 2000 \
        --index data/processed_neural/sparse_paragraph_index_sample.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.common.paragraphs import iter_paragraph_batches

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_neural"
DEFAULT_MODEL = "naver/splade-cocondenser-selfdistil"
DEFAULT_MAX_LENGTH = 256


class SpladeEncoder:
    """Encodes texts into SPLADE sparse vectors, on the GPU when one is available."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, max_length=DEFAULT_MAX_LENGTH):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        # Half precision halves GPU memory use and is faster; CPUs do not benefit.
        self.use_fp16 = torch.device(self.device).type == "cuda"

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    def encode(self, texts):
        """
        Encode texts into SPLADE vectors.

        Args:
            texts (list of str): Paragraphs or queries, already cleaned.

        Returns:
            csr_matrix: One row per text, one column per vocabulary term.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=torch.device(self.device).type,
                                             dtype=torch.float16, enabled=self.use_fp16):
            logits = self.model(**inputs).logits  # [batch, seq_len, vocab]

            # SPLADE max pooling: max over tokens of log(1 + ReLU(logits)), with
            # padding masked out. Done in place, since the logits are the largest
            # tensor here and each copy would take another full-size block of memory.
            weights = logits.relu_().log1p_()
            weights.mul_(inputs["attention_mask"].unsqueeze(-1).to(weights.dtype))
            weights = weights.amax(dim=1)  # [batch, vocab]

        # One transfer per batch, rather than one per nonzero weight.
        return csr_matrix(weights.float().cpu().numpy())


class SparseIndexer:
    """A paragraph x vocabulary matrix of SPLADE weights, searched by dot product."""

    def __init__(self, matrix=None, doc_ids=None, settings=None):
        self.matrix = matrix              # scipy CSC matrix, [num_docs, vocab_size]
        self.doc_ids = doc_ids or []      # row number -> doc_id
        self.settings = settings or {}    # model, max_length, with_title used to encode

    def search(self, query_vector, k=100):
        """
        Return the k highest-scoring paragraphs for an encoded query.

        Args:
            query_vector (csr_matrix or dict): The query's SPLADE vector, as
                one row from SpladeEncoder.encode() or a token_id -> weight dict.
            k (int): Number of results to return.

        Returns:
            list: (doc_id, score) pairs, best first. Paragraphs sharing no term
                with the query score 0 and are left out, so there may be fewer than k.
        """
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []

        if isinstance(query_vector, dict):
            term_ids = np.fromiter(query_vector.keys(), dtype=np.int64)
            term_weights = np.fromiter(query_vector.values(), dtype=np.float32)
        else:
            row = query_vector.tocsr()
            term_ids, term_weights = row.indices, row.data.astype(np.float32)
        if len(term_ids) == 0:
            return []

        # Only the query's terms contribute, so only their columns are read.
        scores = self.matrix[:, term_ids] @ term_weights

        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(-scores[top], kind="stable")]

        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save_index(self, file_path):
        """
        Save the matrix (.npz) and the paragraph IDs and settings (.meta.json).

        Args:
            file_path (str or Path): Path for the .npz file.
        """
        if self.matrix is None or not self.doc_ids:
            raise ValueError("No paragraphs were indexed, so there is nothing to save. "
                             "Check that the collection path points to paragraphs.jsonl.")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(file_path, self.matrix)

        with open(file_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump({"settings": self.settings, "doc_ids": self.doc_ids}, f)

        print(f"Saved index to {file_path} ({self.matrix.nnz:,} nonzero weights)")

    @classmethod
    def load_index(cls, file_path):
        """Load an index saved by save_index()."""
        file_path = Path(file_path)
        with open(file_path.with_suffix(".meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(load_npz(file_path).tocsc(), meta["doc_ids"], meta["settings"])


def count_paragraphs(paragraphs_path):
    """Read the paragraph count from the collection's manifest, or None if there is none."""
    manifest_path = Path(paragraphs_path).with_name("manifest.json")
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["counts"]["paragraphs"]


def build_index(paragraphs_path, model_name=DEFAULT_MODEL, batch_size=64, limit=None,
                device=None, with_title=False):
    """
    Encode every paragraph and build the index.

    Each batch becomes a small sparse matrix, so memory use stays at about
    8 bytes per nonzero weight rather than one Python object per weight.

    Args:
        paragraphs_path (str or Path): Cleaned-text paragraphs.jsonl.
        model_name (str): Hugging Face SPLADE model.
        batch_size (int): Paragraphs per batch; lower it if the GPU runs out of memory.
        limit (int): Only encode the first N paragraphs (for quick tests).
        device (str): "cuda" or "cpu"; defaults to the GPU when there is one.
        with_title (bool): Put the paper title before each paragraph. Use the
            same setting the model was trained with.

    Returns:
        SparseIndexer: The finished index.
    """
    encoder = SpladeEncoder(model_name, device=device)
    total = limit or count_paragraphs(paragraphs_path)

    chunks, doc_ids = [], []
    start = time.perf_counter()
    batches = iter_paragraph_batches(paragraphs_path, batch_size=batch_size,
                                     with_title=with_title, limit=limit)
    for batch_number, (batch_ids, texts) in enumerate(batches, start=1):
        chunks.append(encoder.encode(texts))
        doc_ids.extend(batch_ids)

        if batch_number % 500 == 0:
            elapsed = time.perf_counter() - start
            rate = len(doc_ids) / elapsed
            remaining = f", about {(total - len(doc_ids)) / rate / 60:.0f} min left" if total else ""
            print(f"  {len(doc_ids):,} paragraphs ({rate:.0f}/s{remaining})")

    if not doc_ids:
        raise ValueError(f"No paragraphs were read from {paragraphs_path}.")

    elapsed = time.perf_counter() - start
    print(f"Encoded {len(doc_ids):,} paragraphs in {elapsed / 60:.1f} min "
          f"({len(doc_ids) / elapsed:.0f}/s)")

    # Stored by column, so a search reads only the query terms' columns.
    matrix = vstack(chunks, format="csr")
    chunks.clear()          # free the per-batch pieces before the CSC copy
    matrix = matrix.tocsc()
    settings = {"model": model_name, "max_length": encoder.max_length, "with_title": with_title}
    return SparseIndexer(matrix, doc_ids, settings)


def load_or_build_index(index_path, paragraphs_path, force_reindex=False, **build_options):
    """
    Load a saved index, or build and save it if there is none (or force_reindex is set).

    Args:
        index_path (str or Path): Path of the .npz index file.
        paragraphs_path (str or Path): Collection to build from if needed.
        force_reindex (bool): Rebuild even if a saved index exists.
        **build_options: Passed to build_index().

    Returns:
        SparseIndexer: The loaded or newly built index.
    """
    index_path = Path(index_path)
    if index_path.exists() and not force_reindex:
        print(f"Loading index from {index_path}...")
        return SparseIndexer.load_index(index_path)

    index = build_index(paragraphs_path, **build_options)
    index.save_index(index_path)
    return index


def parse_args():
    parser = argparse.ArgumentParser(description="Build the SPLADE sparse paragraph index.")
    parser.add_argument("--paragraphs", type=Path, default=PROCESSED_DIR / "paragraphs.jsonl",
                        help="Cleaned-text paragraph collection")
    parser.add_argument("--index", type=Path, default=PROCESSED_DIR / "sparse_paragraph_index.npz",
                        help="Where to save the index (use a different path for --limit tests)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face SPLADE model")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Paragraphs per batch (default: 64; lower it if the GPU runs out of memory)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only encode the first N paragraphs (for quick tests)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: GPU if available)")
    parser.add_argument("--with-title", action="store_true",
                        help="Put the paper title before each paragraph")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the index exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_or_build_index(
        args.index,
        args.paragraphs,
        force_reindex=args.force,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        with_title=args.with_title,
    )