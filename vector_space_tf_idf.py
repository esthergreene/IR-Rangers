import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

try: 
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Stopwords set
STOPWORDS = set(stopwords.words('english'))


def tokenize_vsm(text):
    """Clean HTML using BeautifulSoup, extract lowercase tokens, and strip stopwords."""
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    tokens = re.findall(r'\b\w+\b', clean_text.lower())
    return [token for token in tokens if token not in STOPWORDS]


class VectorSpaceTFIDF:
    def __init__(self, index_file_path="vsm_index.json"):
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

    def calculate_document_frequencies(self, doc_term_frequencies):
        """Calculate document frequency (DF) for every unique term across all documents."""
        # Complete this to return dictionary of term: document count
        # calculate_document_frequencies --> sets self.df, term as key, document frequency as value
        df_counts = defaultdict(int)
        for counts in doc_term_frequencies.values():
            for term in counts.keys():
                df_counts[term] += 1
        self.df = dict(df_counts)
        pass

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
        pass

    def vectorize_documents(self, doc_term_frequencies):
        vocab_size = len(self.vocab)
        self.doc_vectors = {}

        for doc_id, counts in doc_term_frequencies.items():
            doc_vector = [0,0] * vocab_size
            for term, count in counts.items():
                if term in self.word_to_idx:
                    pos = self.word_to_idx[term]
                    doc_vector[pos] = self.calculate_tf_idf(count, self.df[term])
            self.doc_vectors[doc_id] = doc_vector
        pass

    def calculate_tf_idf(self, count, df_val):
        if count <= 0 or df_val <= 0 or self.N == 0:
            return 0.0
        tf = 1 + math.log10(count)
        idf = math.log10(self.N / df_val)
        return tf * idf
        pass

    def save_index(self, ):
        pass

    def load_index(self, ):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF Vector Space Model")

    parser.add_argument("--data", "-d", default=r"Answers.json", help="Path to input data JSON file")
    parser.add_argument("--index", "-i", default="Session 7/vsm_index.json", help="Path to saved VSM index file")
    # parser.add_argument("--query", "-q", default="Maine travel guideline ", help="Query string to search")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--force-reindex", action="store_true", help="Force rebuild of index file")

    args = parser.parse_args()

    retriever = VectorSpaceTFIDF(index_file_path=args.index)
    start_index_time = time.perf_counter()
    retriever.index_file(args.data, force_reindex=args.force_reindex)
    index_execution_time = (time.perf_counter() - start_index_time) * 1000
