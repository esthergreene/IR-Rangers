import json
import math
import os
from collections import Counter, defaultdict

from src.preprocessing.tokenizer import tokenize_text

class VectorSpaceTFIDF:
    def __init__(self, corpus_index):
        self.index = corpus_index or {}
        self.N = corpus_index.num_docs
        self.idf = {}
        self.doc_norms = defaultdict(float)

        if self.index and self.N > 0:
            self._precompute_weights_and_norms()

    def _precompute_weights_and_norms(self):
        """Precalculates log-IDF values and Euclidean norm (|D|) for every document."""
        print("Precomputing IDFs and Document Norms...")
        for term, postings in index.vocabulary():
            df = len(postings)
            if df == 0:
                continue

            # Standard log10 IDF calculation
            term_idf = math.log10(self.N / df)
            self.idf[term] = term_idf

            # Support both dict and list postings format
            items = postings.items() if isinstance(postings, dict) else postings

            # Accumulate squared weights for document length normalization
            for doc_id, tf in items:
                weight = (1 + math.log10(tf)) * term_idf
                self.doc_norms[doc_id] += weight * weight

        # Apply final square root to finalize Euclidean norms
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])

            
    def calculate_tf_idf(self, count, df_val):
        if count <= 0 or df_val <= 0 or self.N == 0:
            return 0.0
        tf = 1 + math.log10(count)
        idf = math.log10(self.N / df_val)
        return tf * idf

    # Search function, help from Gemini 3.5 Flash
    def search(self, query_tokens, k=100):
        """Search the document collection using cosine similarity."""
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

        # 2. Accumulate dot product only for docs containing query terms
        for term, q_weight in query_weights.items():
            postings = self.index[term]
            items = postings.items() if isinstance(postings, dict) else postings

            for doc_id, doc_tf in items:
                doc_weight = (1 + math.log10(doc_tf)) * self.idf[term]
                doc_scores[doc_id] += q_weight * doc_weight

        # 3. Calculate Cosine Similarity: dot_product / (query_norm * doc_norm)
        results = []
        for doc_id, dot_product in doc_scores.items():
            d_norm = self.doc_norms.get(doc_id, 0.0)
            if d_norm > 0:
                similarity = dot_product / (query_norm * d_norm)
                results.append((doc_id, similarity))

        # 4. Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
        
# Should want to save intergers, save term frequency values, and use that to calucate the scores, 
# instead of saving the tf-idf values. 
# # This will allow you to change the scoring function without having to re-index the data.
