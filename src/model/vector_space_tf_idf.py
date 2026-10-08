import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

# Stopwords set, written with help from Gemini 3.5 Flash
def load_stopwords(filepath="./config/stopwords.txt"):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:


        class VectorSpaceTFIDF:
            def __init__(self, index_file_path="./src/preprocessing/index.json"):
                self.index_file_path = index_file_path
                # List of unique terms (size |V|) -- We need this to bulid the vocabulary
                self.vocab = []
                # Mapping: word → vector position index -- What word each index of a vector is representing
                self.word_to_idx = {}
                # doc_id → list of floats of length |V| -- Mapping of documents to their vector
                self.doc_vectors = {}
                # doc_id → raw document dictionary -- Used to show the results
                self.documents = {}
                # term → document frequency -- How often a terms appears in a documents/answers
                self.df = {}
                # Total document count
                self.N = 0

            def calculate_document_frequencies(self, doc_term_frequencies):
                """Calculate document frequency (DF) for every unique term across all documents."""
                # Complete this to return dictionary of term: document count
                # calculate_document_frequencies --> sets self.df, term as key, document frequency as value
                df_counts = defaultdict(int)
                for counts in doc_term_frequencies.values():
                    for term in counts.keys():
                        df_counts[term] += 1
                self.df = dict(df_counts)

            def build_vocabulary(self, term_frequencies, max_vocab_size=20000):
                """Rank unique corpus words by total collection frequency and retain top-K terms."""
                # Build vocabulary --> sets self.vocab and self.word_to_idx (what term does each postion represent)

                collection_frequencies = Counter()
                for counts in term_frequencies.values():
                    collection_frequencies.update(counts)

                if max_vocab_size and max_vocab_size < len(collection_frequencies):
                    top_k_terms = [term for term, _ in collection_frequencies.most_common(max_vocab_size)]
                else:
                    top_k_terms = list(collection_frequencies.keys())

                self.vocab = top_k_terms
                self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

            def vectorize_documents(self, doc_term_frequencies):
                vocab_size = len(self.vocab)
                self.doc_vectors = {}

                for doc_id, counts in doc_term_frequencies.items():
                    doc_vector = [0.0] * vocab_size
                    for term, count in counts.items():
                        if term in self.word_to_idx:
                            pos = self.word_to_idx[term]
                            doc_vector[pos] = self.calculate_tf_idf(count, self.df[term])
                    self.doc_vectors[doc_id] = doc_vector
            
            # Loading the index file, help from Gemini 3.5 Flash
            def index_file(self, data_path, force_reindex=False):
                # Keep track of term frequencies in each document: doc_term_frequencies
                    # Doc_id --> {term: frequency}
                # Keep track of unique words: unique_words
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
                    self.documents[doc_id] = doc
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
                pass

            # Saving the index file, help from Gemini 3.5 Flash
            def save_index(self):
                os.makedirs(os.path.dirname(self.index_file_path) or ".", exist_ok=True)
                index_data = {
                    "N": self.N,
                    "vocab": self.vocab,
                    "df": self.df,
                    "doc_vectors": self.doc_vectors,
                    "documents": self.documents
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
                self.documents = index_data["documents"]

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

            def cosine_similarity(self, vec1, vec2):
                """Calculate cosine similarity between two dense vectors."""
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = math.sqrt(sum(a * a for a in vec1))
                norm2 = math.sqrt(sum(b * b for b in vec2))

                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot_product / (norm1 * norm2)

            def search(self, query_tokens, top_k=5):
                """Search the document collection using cosine similarity."""
                query_vec = self.vectorize_query(query_tokens)
                scores = []

                for doc_id, doc_vec in self.doc_vectors.items():
                    score = self.cosine_similarity(query_vec, doc_vec)
                    if score > 0:
                        scores.append((doc_id, score))

                # Sort by similarity score in descending order
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="TF-IDF Vector Space Model")

        parser.add_argument("--data", "-d", default=r"Train.json", help="Path to input data JSON file",)
        parser.add_argument("--index", "-i", default="Session 7/vsm_index.json", help="Path to saved VSM index file",)
        parser.add_argument("--stopwords", "-s", default="stopwords.txt", help="Path to custom stopwords file",)
        parser.add_argument("--query", "-q", default="Maine travel guideline", help="Query string to search",)
        parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of top results to return",)
        parser.add_argument("--force-reindex", action="store_true", help="Force rebuild of index file",)

        args = parser.parse_args()

        # Load custom stopwords
        stopwords = load_stopwords(args.stopwords)

        retriever = VectorSpaceTFIDF(index_file_path=args.index)
        start_index_time = time.perf_counter()

        retriever.index_file(args.data, force_reindex=args.force_reindex)
        index_execution_time = (time.perf_counter() - start_index_time) * 1000
        print(f"Indexing completed in {index_execution_time:.2f} ms")

        # Example query execution (Simple whitespace tokenization & lowercase filtering)
        if args.query:
            query_tokens = [
                word.lower()
                for word in args.query.split()
                if word.lower() not in stopwords
            ]
            results = retriever.search(query_tokens, top_k=args.top_k)

            print(f"\nTop {args.top_k} results for query: '{args.query}'")
            for doc_id, score in results:
                doc_info = retriever.documents.get(doc_id, {})
                print(f"Doc ID: {doc_id} | Score: {score:.4f}")

    # Should want to save intergers, save term frequency values, and use that to calucate the scores, 
    # instead of saving the tf-idf values. 
    # This will allow you to change the scoring function without having to re-index the data.
