from pathlib import Path
import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.index.sparse_indexer import SparseIndexer, SpladeEncoder 
from src.common.paragraphs import load_paragraph_texts

class SparseSearcher:
    """Combines the SpladeEncoder and SparseIndexer to execute queries."""
    def __init__(self, index_path, model_name="naver/splade-cocondenser-selfdistil"):
        print(f"Loading search index from {index_path}...")
        self.index = SparseIndexer.load_index(index_path)
        
        print(f"Initializing encoder ({model_name})...")
        self.encoder = SpladeEncoder(model_name=model_name)

    def search(self, query_text, k=5):
        """
        Encodes a query string and searches the index for top-k results.
        
        Args:
            query_text (str): The search query typed by the user.
            k (int): Number of top results to return.
            
        Returns:
            list of dict: Top results containing doc_id, score, and text.
        """
        # Step 1: Encode the single query string into a list of sparse dicts
        # Using our SpladeEncoder, the same Encoder used for indexing paragraphs
        sparse_query_list = self.encoder.encode([query_text])
        query_vector = sparse_query_list[0] # Extract the first (and only) query dict

        return self.index.search(query_vector, k=k)
        # # Step 2: Search the index using matrix multiplication and argpartition
        # raw_results = self.index.search(query_vector, k=k) # Returns [(doc_id, score), ...]
        
        # if not raw_results:
        #     return []

        # # Extract just the doc_ids to fetch their text
        # doc_ids = [doc_id for doc_id, score in raw_results]
        
        # # Load the matching paragraph text from your corpus file
        # repo_root = Path(__file__).resolve().parents[2]
        # paragraphs_path = repo_root / "data" / "processed_neural" / "paragraphs.jsonl"
        # text_mapping = load_paragraph_texts(paragraphs_path, doc_ids=doc_ids)

        # # Format final results
        # formatted_results = []
        # for doc_id, score in raw_results:
        #     formatted_results.append({
        #         "doc_id": doc_id,
        #         "score": score,
        #         "text": text_mapping.get(doc_id, "Text not found")
        #     })

        # return formatted_results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed_neural"
    index_path = processed_dir / "sparse_paragraph_index.npz"

    # Initialize the searcher
    searcher = SparseSearcher(index_path=index_path)

    # Run a test query
    test_query = "How do sparse neural retrieval models expand query terms?"
    print(f"\nRunning search for query: '{test_query}'\n" + "-"*50)

    results = searcher.search(test_query, k=3)

    for idx, res in enumerate(results, 1):
        print(f"Rank {idx} | Doc ID: {res['doc_id']} | Score: {res['score']:.4f}")
        print(f"Text: {res['text']}\n" + "-"*50)