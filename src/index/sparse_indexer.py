import json
from collections import defaultdict
from pathlib import Path

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    print("Saved index to: ", file_path)


class SparseIndexer:
    def __init__(self):
        self.inverted_index = defaultdict(list)

    def build_index(self, encoded_documents):
        """
        Build an inverted index from the encoded documents.

        Args:
            encoded_documents (dict): A dictionary where keys are document IDs and values are sparse vectors.
        """
        for doc_id, sparse_vector in encoded_documents.items():
            for token_id, weight in sparse_vector.items():
                if weight > 0:
                    self.inverted_index[token_id].append((doc_id, weight))

    def save_index(self, file_path):
        """
        Save the inverted index to a JSON file.

        Args:
            file_path (str): The path to the file where the index will be saved.
        """
        save_json(self.inverted_index, file_path)

    @classmethod
    def from_dict(cls, data):
        """Reconstruct a SparseIndexer from a loaded dictionary."""
        instance = cls()
        # Convert standard dict back to defaultdict of lists
        instance.inverted_index = defaultdict(list, data)
        return instance


def load_index(file_path, force_reindex=False):
    file_path = Path(file_path)
    
    if force_reindex or not file_path.exists():
        collection_name = {
            "sparse_paper_index.json": "sparse_papers.jsonl",
            "sparse_paragraph_index.json": "sparse_paragraphs.jsonl"
        }

        collection_path = file_path.with_name(
            collection_name[file_path.name]
        )

        return build_and_save_index(collection_path, file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return SparseIndexer.from_dict(data)


def build_index_from_jsonl(file_path):
    index = SparseIndexer()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            doc_id = record.get("doc_id")
            # Adjust the key here if your JSONL uses something other than 'vector' or 'sparse_vector'
            sparse_vector = record.get("vector") or record.get("sparse_vector")

            if doc_id and sparse_vector:
                index.build_index({doc_id: sparse_vector})

    return index


def build_and_save_index(collection_path, index_path):
    """Build an index from a collection and then save it to eventually return it."""
    index = build_index_from_jsonl(collection_path)
    index.save_index(index_path)

    return index


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"

    # Build the paper index from the preprocessed collection.
    print("Building sparse paper index...")
    paper_index = build_and_save_index(
        processed_dir / "sparse_papers.jsonl",
        processed_dir / "sparse_paper_index.json"
    )

    del paper_index

    # Build the paragraph index from the preprocessed collection.
    print("Building sparse paragraph index...")
    paragraph_index = build_and_save_index(
        processed_dir / "sparse_paragraphs.jsonl",
        processed_dir / "sparse_paragraph_index.json"
    )