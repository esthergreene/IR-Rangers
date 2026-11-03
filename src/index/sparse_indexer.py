from pathlib import Path
import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.common.paragraphs import iter_paragraph_batches 

class SpladeEncoder:
    def __init__(self, model_name_or_path="naver/splade-cocondenser-selfdistil", device="cpu"): # Running on CPU because GPU memory kept running out
        self.device = device
        print(f"Loading SPLADE model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def encode(self, texts, with_title=False):
        """
        Encodes a list of texts into sparse vector dictionaries (token_id -> weight).
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # SPLADE aggregation: max over sequence length of log(1 + ReLU(logits))
        # (Exact implementation details depend on the specific SPLADE variant you use)
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        # Max pooling across the attention mask length dimension
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        weights = torch.max(relu_log * attention_mask, dim=1).values  # Shape: [batch_size, vocab_size]

        sparse_vectors = []
        # Extract non-zero weights per document in the batch
        nz_indices = torch.nonzero(weights, as_tuple=False)
        
        # Group non-zeros by batch item
        batch_dict = {i: {} for i in range(len(texts))}
        for row, col in nz_indices:
            weight = weights[row, col].item()
            if weight > 0:
                batch_dict[row.item()][col.item()] = weight

        for i in range(len(texts)):
            sparse_vectors.append(batch_dict[i])

        return sparse_vectors


def build_index_with_encoder(paragraphs_path, model_name_or_path="naver/splade-cocondenser-selfdistil", batch_size=16, limit=None):
    """
    Streams paragraphs from jsonl, encodes them in batches, and builds the SparseIndexer.
    """
    index = SparseIndexer()
    encoder = SpladeEncoder(model_name_or_path)
    
    docs = {}
    print("Encoding paragraphs and building index...")
    
    # Changed with_title=True to with_title=False since records lack a 'title' key
    for doc_ids, texts in iter_paragraph_batches(paragraphs_path, batch_size=batch_size, with_title=False, limit=limit):
        # Encode batch into list of sparse dicts
        sparse_vectors = encoder.encode(texts)
        
        for doc_id, s_vec in zip(doc_ids, sparse_vectors):
            docs[doc_id] = s_vec

    print(f"Total encoded documents loaded: {len(docs)}")
    index.build_index(docs)
    return index

def load_or_build_index_safe(file_path, collection_path=None, force_reindex=False, model_name=None):
    file_path = Path(file_path)
    
    if force_reindex or not file_path.exists():
        if collection_path is None:
            raise ValueError("Collection path required to build index.")
        
        index = build_index_with_encoder(collection_path, model_name_or_path=model_name)
        index.save_index(file_path)
        return index

    return SparseIndexer.load_index(file_path)

class SparseIndexer:
    def __init__(self):
        self.matrix = None 
        self.doc_ids = [] 
        self.doc_id_to_row = {}

    def build_index(self, encoded_documents):
        """
        Build a CSR matrix index from encoded documents efficiently using arrays.
        
        Args:
            encoded_documents (dict): A dictionary where keys are document IDs 
                                      and values are sparse vectors (token_id -> weight).
        """
        rows = []
        cols = []
        data = []

        for row_idx, (doc_id, sparse_vector) in enumerate(encoded_documents.items()):
            self.doc_ids.append(doc_id)
            self.doc_id_to_row[doc_id] = row_idx

            for token_id, weight in sparse_vector.items():
                if weight > 0:
                    rows.append(row_idx)
                    cols.append(int(token_id))
                    data.append(float(weight))

        if rows:
            # Determine column dimension dynamically based on max token ID encountered
            num_cols = max(cols) + 1 if cols else 0
            self.matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(self.doc_ids), num_cols), 
                dtype=np.float32
            )
        else:
            self.matrix = csr_matrix((0, 0), dtype=np.float32)

    def search(self, query_vector, k=10):
        """
        Search the index using matrix multiplication and top-k argpartition.
        
        Args:
            query_vector (dict or np.ndarray): Token weights for the query.
            k (int): Number of top results to return.
            
        Returns:
            list of tuples: (doc_id, score) sorted descending by score.
        """
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []

        # Convert query dict to a dense vector aligned with matrix columns
        if isinstance(query_vector, dict):
            q_arr = np.zeros(self.matrix.shape[1], dtype=np.float32)
            for tid, weight in query_vector.items():
                if tid < self.matrix.shape[1]:
                    q_arr[tid] = float(weight)
        else:
            q_arr = query_vector

        # Matrix-vector multiplication -> scores per document row
        scores = self.matrix @ q_arr

        # Handle case where k is larger than the number of documents
        k = min(k, len(self.doc_ids))
        if k <= 0:
            return []

        # Use argpartition to find top k indices efficiently
        # argpartition doesn't fully sort, but puts the top k elements at the end
        top_k_indices = np.argpartition(scores, -k)[-k:]
        # Sort the top k indices explicitly by their scores descending
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score > 0: 
                results.append((self.doc_ids[idx], score))

        return results

    def save_index(self, file_path):
        """
        Save the CSR matrix using save_npz and companion metadata for doc_ids.
        
        Args:
            file_path (str or Path): Base path to save the index (.npz).
        """
        file_path = Path(file_path)
        save_npz(file_path, self.matrix)

        # Save companion metadata (doc_ids mapping)
        meta_path = file_path.with_suffix(".meta.json")
        if len(self.doc_ids) == 0:
            raise ValueError("Should not happen!")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        print(f"Saved index to: {file_path} and {meta_path}")

    @classmethod
    def load_index(cls, file_path):
        """Load SparseIndexer from disk (.npz matrix + .meta.json)."""
        file_path = Path(file_path)
        instance = cls()
        
        instance.matrix = load_npz(file_path)

        meta_path = file_path.with_suffix(".meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            instance.doc_ids = json.load(f)

        instance.doc_id_to_row = {doc_id: i for i, doc_id in enumerate(instance.doc_ids)}
        return instance


def build_index_from_jsonl(file_path):
    """Builds the index by streaming documents to avoid loading everything into a tuple list."""
    index = SparseIndexer()
    docs = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            doc_id = record.get("doc_id")
            sparse_vector = record.get("sparse_vector")
            if doc_id and sparse_vector:
                docs[doc_id] = {int(k): v for k, v in sparse_vector.items()}

    index.build_index(docs)
    return index


def load_or_build_index(file_path, collection_path=None, force_reindex=False):
    file_path = Path(file_path)
    
    if force_reindex or not file_path.exists():
        if collection_path is None:
            raise ValueError("Collection path required to build index.")
        
        index = build_index_from_jsonl(collection_path)
        index.save_index(file_path)
        return index

    return SparseIndexer.load_index(file_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed_neural"
    
    index_path = processed_dir / "sparse_paragraph_index.npz"
    collection_path = processed_dir / "paragraphs.jsonl"

    print("Loading or building sparse paragraph index...")
    paragraph_index = load_or_build_index_safe(
        file_path=index_path, 
        collection_path=collection_path, 
        model_name="naver/splade-cocondenser-selfdistil",
        force_reindex=True
    )