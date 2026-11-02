"""
This file handles our dense retrieval model using Sentence-BERT and FAISS.

Sentence-BERT converts queries into embeddings, and FAISS helps us
find the documents that are most similar to each query.

We can use either a pretrained model or one that we fine-tune ourselves.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

class DenseModel:
    """
    Initialize the Sentence-BERT model.

    Args:
         model_path: Pretrained or fine-tuned Sentence-BERT model.
    """
    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)

    def encode_query(self, query):
        """
        This converts a query into an embedding using Sentence-BERT
         
         Args:
        query: The OG query text.
                
        Returns:
            A normalized numpy array containing the query embedding.
        """
        embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        return np.asarray(embedding, dtype=np.float32)

    def encode_documents(self, documents, batch_size=32):
        """
        This converts a list of documents into embeddings using our Sentence-BERT
        
        Args:
            documents: A list of doument texts.
            bath_size: How many documents to encode at once.
            
        Returns:
            A numpy array containing the normalized document embeddings.
        """
        embeddings = self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        return np.asarray(embeddings, dtype=np.float32)

    def build_index(self, embeddings, doc_ids):
        """
        This creates a FAISS index using our document embeddings.
        
        Args:
            embeddings: The document embeddings.
            doc_ids: The original document IDS.
            
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError("Each embedding needs a matching document ID.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.doc_ids = list(doc_ids)

    def search(self, query, k=100):
        """
        This searches our FAISS index for documents similar to a query.
        
        Args:
            query: The original query text.
            k: The maximum number of documents to return.
            
        Returns:
            A list of document IDs and their similarity scores.
        """
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        results = []

        for index, score in zip(indices[0], scores[0]):
            if index == -1:
                continue

            doc_id = self.doc_ids[index]
            results.append((doc_id, float(score)))

        return results

    def load_index(self, index_path, ids_path):
        """
        This loads a saved FAISS index and its document IDs.

        Args:
            index_path: The path to our saved FAISS index.
            ids_path: The path to our saved document IDs.
        """
        self.index = faiss.read_index(str(index_path))

        with open(ids_path, "r", encoding="utf-8") as file:
            self.doc_ids = json.load(file)
        