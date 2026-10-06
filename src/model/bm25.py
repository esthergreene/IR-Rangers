"""
BM25 retrieval model.

Scores documents against a query using Okapi BM25 with a non-negative IDF
(the "Lucene accurate" variant in Kamphuis et al., ECIR 2020):

    score(q, d) = sum over terms t in q of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))

    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

The model works on tokens, not text. Callers tokenize queries with the same
saved preprocessing config that was used to build the index.

Usage:
    from src.model.bm25 import BM25Model

    model = BM25Model(corpus_index, k1=0.9, b=0.4)
    results = model.search(query_tokens, k=100)
"""
import heapq
import math
from collections import defaultdict

class BM25Model:
    """
    BM25 ranking over a pre-built inverted index.

    Expected index interface (to confirm with the indexing code):
        num_docs            N, the number of documents in the collection
        avg_doc_length      average document length in tokens
        doc_lengths         length in tokens of each document
        vocabulary()        every term in the index
        df(term)            number of documents containing the term
        postings(term)      (doc_ids, term_frequencies) for documents containing the term

    IDF values are computed once, when the model is created. Length
    normalizations are recomputed whenever k1 or b change.
    """
    def __init__(self, corpus_index, k1=0.9, b=0.4):
        """
        Args:
            corpus_index: Inverted index for one collection (papers or paragraphs).
            k1 (float): Term frequency saturation. Higher values let repeated
                terms keep adding to the score for longer. Must be 0 or more.
            b (float): Document length normalization, from 0 (none) to 1 (full).
        """
        self.corpus_index = corpus_index
        self.idf = self._compute_idf()
        self.set_params(k1, b)

    def set_params(self, k1, b):
        """
        Change k1 and b, and recompute the length normalizations that depend on them.

        Cheaper than creating a new model during tuning, since IDF does not
        depend on k1 or b and is not recomputed.

        Args:
            k1 (float): Must be 0 or more.
            b (float): Must be between 0 and 1.

        Raises:
            ValueError: If k1 or b is out of range.
        """
        self._validate_params(k1, b)
        self.k1 = k1
        self.b = b
        self.length_norm = self._compute_length_norm()

    def search(self, query_tokens, k=100):
        """
        Return the k highest-scoring documents for a query.

        Only documents containing at least one query term are scored. Query terms not 
        in the index are skipped. A term repeated in the query is counted each time it appears.

        Args:
            query_tokens (list of str): Query, already tokenized with the index's preprocessing config.
            k (int): Maximum number of results to return.
        
        Returns:
            list of (doc_id, score): Up to k (doc_id, score) pairs, sorted descending by score. Equal
            scores are ordered by internal document number so runs are reproducible. Empty if no query term is in the index.

            doc_id(i): the document ID string for internal document number i
        """
        scores = defaultdict(float)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None: # term not in the index
                continue
            doc_numbers, term_frequencies = self.corpus_index.postings(term)
            for doc_number, tf in zip(doc_numbers, term_frequencies):
                scores[doc_number] += self._term_score(idf, tf, doc_number)
            
        # Highest score first; equal scores ordered by internal document number.
        top = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.corpus_index.doc_id(doc_number), score) for doc_number, score in top]
    
    def score(self, query_tokens, doc_id):
        """
        Compute the BM25 score of a single document for a query.

        Slower than search(), since it looks up each term separately. Meant for tests and debugging, e.g.,
        checking a score worked out manually.

        Args:
            query_tokens (list of str): Query, already tokenized.
            doc_id: The document to score.
        
        Returns:
            float: The document's score; 0.0 if it contains no query terms.
        """
        target = self.corpus_index.doc_number(doc_id)
        total = 0.0
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None: # term not in the index
                continue
            doc_numbers, term_frequencies = self.corpus_index.postings(term)
            for doc_number, tf in zip(doc_numbers, term_frequencies):
                if doc_number == target:
                    total += self._term_score(idf, tf, doc_number)
        return total
    
    def _term_score(self, idf, tf, doc_number):
        """
        BM25 contribution of one query term to one document.

        Args:
            idf (float): The term's IDF.
            tf (int): How many times the term appears in the document.
            doc_number (int): The document's internal number.
        
        Returns:
            float: idf * tf * (k1 + 1) / (tf + length normalization).
        """
        return idf * tf * (self.k1 + 1) / (tf + self.length_norm[doc_number])
    
    
    def _compute_idf(self):
        """
        Compute the IDF of every term in the index.

        Uses log(1 + (N - df + 0.5) / (df + 0.5)), which stays positive even for terms
        in more than half the documents.

        Returns:
            dict: term -> IDF.
        """
        num_docs = self.corpus_index.num_docs
        idf = {}
        for term in self.corpus_index.vocabulary():
            df = self.corpus_index.df(term)
            idf[term] = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        return idf

    def _compute_length_norm(self):
        """
        Compute each document's length normalization for the current k1 and b.

        This is the document-only part of the BM25 denominator,
        k1 * (1 - b + b * len(d) / avg_len), computed once per document instead of
        once per document per query term.

        Returns:
            Per-document values, in the same order as the index's doc_lengths.
        """
        avg_length = self.corpus_index.avg_doc_length
        return [
            self.k1 * (1 - self.b + self.b * length / avg_length)
            for length in self.corpus_index.doc_lengths
        ]
    
    @staticmethod
    def _validate_params(k1, b):
        """
        Check that k1 and b are in range.

        Raises:
            ValueError: If k1 is negative or b is outside [0, 1].
        """
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(
                f"Invalid BM25 parameters: k1 must be 0 or more and b must be "
                f"between 0 and 1; got k1={k1}, b={b}."
            )