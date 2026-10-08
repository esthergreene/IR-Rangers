"""TF-IDF cosine-similarity retrieval over a pre-built inverted index.

The index is expected to provide ``num_docs``, ``vocabulary()``, ``df(term)``,
``postings(term)`` (parallel lists of document numbers and term frequencies),
and ``doc_id(doc_number)``. Queries must be tokenized with the same
preprocessing configuration used to build the index.

Usage:
    from src.model.tfidf import TFIDFModel

    model = TFIDFModel(corpus_index)
    results = model.search(query_tokens, k=100)
"""
import heapq
import math
from collections import Counter, defaultdict

class TFIDFModel:
    """Rank indexed documents by TF-IDF cosine similarity."""

    def __init__(self, corpus_index):
        """Create a TF-IDF model for ``corpus_index``.

        IDF values and document vector norms are computed once at
        initialization.

        Args:
            corpus_index: Inverted index for one collection (papers or paragraphs).
        """
        self.index = corpus_index
        self.N = self.index.num_docs
        self.idf = {}
        self.doc_norms = [0.0] * self.N

        if self.index and self.N > 0:
            self._precompute_weights_and_norms()

    def _precompute_weights_and_norms(self):
        """Compute log-IDF values and Euclidean norms for all document vectors."""
        print("Precomputing IDFs and Document Norms...")
        for term in self.index.vocabulary():
            df = self.index.df(term)
            if df == 0:
                continue

            # Standard log10 IDF calculation
            term_idf = math.log10(self.N / df)
            self.idf[term] = term_idf

            # Accumulate squared weights for document length normalization
            for doc_number, tf in zip(*self.index.postings(term)):
                weight = (1 + math.log10(tf)) * term_idf
                self.doc_norms[doc_number] += weight * weight

        # Apply final square root to finalize Euclidean norms
        for doc_number, norm_sq in enumerate(self.doc_norms):
            self.doc_norms[doc_number] = math.sqrt(norm_sq)

    def calculate_tf_idf(self, count, df_val):
        """Return the TF-IDF weight for a term frequency and document frequency."""
        if count <= 0 or df_val <= 0 or self.N == 0:
            return 0.0
        tf = 1 + math.log10(count)
        idf = math.log10(self.N / df_val)
        return tf * idf

    def search(self, query_tokens, k=100):
        """Return the ``k`` highest-scoring documents for a tokenized query.

        Query terms not in the index are skipped, and repeated terms contribute
        according to their query frequency. Only documents containing at least
        one indexed query term are scored. Ties are ordered by internal document
        number for deterministic results.

        Args:
            query_tokens: Query tokens processed with the index's tokenizer.
            k: Maximum number of results to return.

        Returns:
            Up to ``k`` ``(doc_id, score)`` pairs in descending score order.
        """
        if not query_tokens or not self.index:
            return []

        query_counts = Counter(query_tokens)
        query_weights = {}
        query_norm_sq = 0.0

        # 1. Calculate query weights and total query vector norm
        for term, count in query_counts.items():
            if term in self.idf and self.idf[term] > 0:
                tf_q = 1 + math.log10(count)
                weight_q = tf_q * self.idf[term]
                query_weights[term] = weight_q
                query_norm_sq += weight_q * weight_q
        if not query_weights or query_norm_sq == 0:
            return []

        query_norm = math.sqrt(query_norm_sq)
        doc_scores = defaultdict(float)

        for term, q_weight in query_weights.items():
            for doc_number, doc_tf in zip(*self.index.postings(term)):
                doc_weight = (1 + math.log10(doc_tf)) * self.idf[term]
                doc_scores[doc_number] += q_weight * doc_weight

        # Keep only the best k candidates instead of sorting every match.
        top = heapq.nsmallest(
            k,
            (
                (
                    doc_number,
                    dot_product / (query_norm * self.doc_norms[doc_number]),
                )
                for doc_number, dot_product in doc_scores.items()
                if self.doc_norms[doc_number] > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            (self.index.doc_id(doc_number), score)
            for doc_number, score in top
        ]
