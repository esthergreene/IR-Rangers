import json
import math
import os
from collections import Counter, defaultdict

# Stopwords set, written with help from Gemini 3.5 Flash
def load_stopwords(filepath="./config/stopwords.txt"):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


class VectorSpaceTFIDF:
    def __init__(self, inverted_index = none, N=0):
        self.index = inverted_index or {}
        self.N = 0
        self.idf = {}
        self.doc_norms = deafaultdict(float)

        if self.index and self.N > 0:
            self._precompute_idf_and_norms()

    def _precompute_weights_and_norms(self):
        """Precalculates log-IDF values and Euclidean norm (|D|) for every document."""
        print("Precomputing IDFs and Document Norms...")
        for term, postings in self.index.items():
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
                if tf > 0 and term_idf > 0:
                    weight = (1 + math.log10(tf)) * term_idf
                    self.doc_norms[doc_id] += weight * weight

        # Apply final square root to finalize Euclidean norms
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])

    def load_from_index_file(self, index_file_path, num_documents=None):
        """Load pre-built inverted index from JSON and compute norms."""
        print(f"Loading inverted index from {index_file_path}...")
        with open(index_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "index" in data:
            self.index = data["index"]
            self.N = num_documents or data.get("N", len(data.get("documents", [])))
        else:
            self.index = data
            if num_documents is None:
                # Infer total N by union of all document IDs across postings
                all_docs = set()
                for postings in self.index.values():
                    items = postings.keys() if isinstance(postings, dict) else [p[0] for p in postings]
                    all_docs.update(items)
                self.N = len(all_docs)
            else:
                self.N = num_documents

        self._precompute_weights_and_norms()
            
    # Loading the index file, help from Gemini 3.5 Flash
    def index_file(self, data_path, force_reindex=False):
        # Keep track of term frequencies in each document: doc_term_frequencies
        # Doc_id --> {term: frequency}
        # Kwep track of unique words: unique_words
        """Index raw documents by executing the modular VSM pipeline."""
        if os.path.exists(self.index_file_path) and not force_reindex:
            print(f"Index file '{self.index_file_path}' found. Loading from disk...")
            self.load_index()
            return

        print(f"Indexing source file '{data_path}'...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.N = len(data)
        # docid as key, dict "term: frequency" as value
        doc_term_frequencies = {}
        unique_words = set()

        for doc in data:
            doc_id = doc["Id"]
            # For results represenation
            tokens = tokenize_vsm(doc["Text"])
            ##
            # Complete this part to update doc_term_frequencies and unique words
            ##
            doc_term_frequencies[doc_id] = Counter(tokens)
            unique_words.update(tokens)

        # Calculates document frequency for each term
        self.calculate_document_frequencies(doc_term_frequencies)
        # Build vocabulary and mapping of vocabularies to indicies
        self.build_vocabulary(doc_term_frequencies)

        # Do this last; you may event comment it out for now
        self.vectorize_documents(doc_term_frequencies)

        print(f"Vocabulary size |V|: {len(self.vocab)} unique terms across {self.N} documents.")
        self.save_index()

    def calculate_tf_idf(self, count, df_val):
        if count <= 0 or df_val <= 0 or self.N == 0:
            return 0.0
        tf = 1 + math.log10(count)
        idf = math.log10(self.N / df_val)
        return tf * idf

    # Saving the index file, help from Gemini 3.5 Flash
    def save_index(self):
        os.makedirs(os.path.dirname(self.index_file_path) or ".", exist_ok=True)
        index_data = {
            "N": self.N,
            "vocab": self.vocab,
            "df": self.df,
            "doc_vectors": self.doc_vectors,
        }
        with open(self.index_file_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
        print(f"Index successfully saved to {self.index_file_path}")

    # Loading the index file, help from Gemini 3.5 Flash
    def load_index(self):
        with open(self.index_file_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.N = index_data["N"]
        self.vocab = index_data["vocab"]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.df = index_data["df"]
        self.doc_vectors = index_data["doc_vectors"]

    def vectorize_query(self, query_tokens):
        """Convert query tokens into a TF-IDF vector matching the vocabulary space."""
        vocab_size = len(self.vocab)
        query_vector = [0.0] * vocab_size
        query_counts = Counter(query_tokens)

        for term, count in query_counts.items():
            if term in self.word_to_idx:
                pos = self.word_to_idx[term]
                df_val = self.df.get(term, 0)
                query_vector[pos] = self.calculate_tf_idf(count, df_val)

        return query_vector

    # Search function, help from Gemini 3.5 Flash
    def search(self, query_tokens, top_k=100):
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
                if doc_tf > 0:
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
        return results[:top_k]

    if __name__ == "__main__":
        

    # Should want to save intergers, save term frequency values, and use that to calucate the scores, 
    # instead of saving the tf-idf values. 
    # This will allow you to change the scoring function without having to re-index the data.
