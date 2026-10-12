"""
This file handles building the inverted indexes for our paper and
paragraph collections.

Instead of reading and processing all the original papers again,
the indexer uses the preprocessed JSONL files that we already created.
Each document has an ID and a list of tokens that we can use to
build the index.

For each document, we keep track of which terms appear, how often
they appear, and which document they came from. We also calculate
document lengths and the average length across the collection.

All of this information gets stored in posting lists so our BM25
and TF-IDF models can use it when searching for relevant documents.

To build both indexes, run:
    python -m src.index.indexer

The finished indexes are saved in data/processed/.
"""

import json
from collections import Counter

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    print("Saved index to: ", file_path)

class CorpusIndex:
    def __init__(self):
        self.num_docs = 0
        self.avg_doc_length = 0.0
        self.doc_lengths = []

        self._postings = {}
        self._doc_ids = []
        self._doc_numbers = {}

    def vocabulary(self):
        return self._postings.keys()

    def df(self, term):
        if term not in self._postings:
            return 0

        return len(self._postings[term][0])

    def postings(self, term):
        if term not in self._postings:
            return [], []

        return self._postings[term]

    def doc_id(self, doc_number):
        return self._doc_ids[doc_number]

    def doc_number(self, doc_id):
        return self._doc_numbers[doc_id]

    def to_dict(self):
        return {
            "num_docs": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "postings": self._postings,
            "doc_ids": self._doc_ids
        }

    @classmethod
    def from_dict(cls, data):
        index = cls()
        index.num_docs = data["num_docs"]
        index.avg_doc_length = data["avg_doc_length"]
        index.doc_lengths = data["doc_lengths"]
        index._postings = data["postings"]
        index._doc_ids = data["doc_ids"]
        index._doc_numbers = {
            doc_id: doc_number
            for doc_number, doc_id in enumerate(index._doc_ids)
        }

        return index

def load_index(file_path, force_reindex=False):
    if force_reindex:
        from pathlib import Path

        index_path = Path(file_path)

        collection_name = {
            "paper_index.json": "papers.jsonl",
            "paragraph_index.json": "paragraphs.jsonl"
        }

        collection_path = index_path.with_name(
            collection_name[index_path.name]
        )

        return build_and_save_index(collection_path, index_path)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return CorpusIndex.from_dict(data)

def build_index_from_jsonl(file_path):
    """So here I am building an inverted index from a preprocessed JSONL collection. Fingers crossed."""
    index = CorpusIndex()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            doc_id = record["doc_id"]
            tokens = record["tokens"]
            doc_number = index.num_docs

            index._doc_ids.append(doc_id)
            index._doc_numbers[doc_id] = doc_number
            index.doc_lengths.append(len(tokens))

            term_frequencies = Counter(tokens)

            for term, frequency in term_frequencies.items():
                if term not in index._postings:
                    index._postings[term] = ([], [])

                index._postings[term][0].append(doc_number)
                index._postings[term][1].append(frequency)

            index.num_docs += 1

    if index.num_docs > 0:
        index.avg_doc_length = sum(index.doc_lengths) / index.num_docs

    return index

def build_and_save_index(collection_path, index_path):
    """Build an index from a collection and then save it to eventually return it."""
    index = build_index_from_jsonl(collection_path)
    save_json(index.to_dict(), index_path)

    return index

if __name__ == "__main__":
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"

    # Build the paper index from the preprocessed collection.
    print("Building paper index...")
    paper_index = build_and_save_index(
        processed_dir / "papers.jsonl",
        processed_dir / "paper_index.json"
    )

    print("Papers indexed:", paper_index.num_docs)
    print("Unique terms in paper index:", len(paper_index.vocabulary()))
    print("Average paper length:", paper_index.avg_doc_length)

    del paper_index

    # Build the paragraph index from the preprocessed collection.
    print("Building paragraph index...")
    paragraph_index = build_and_save_index(
        processed_dir / "paragraphs.jsonl",
        processed_dir / "paragraph_index.json"
    )

    print("Paragraphs indexed:", paragraph_index.num_docs)
    print("Unique terms in paragraph index:", len(paragraph_index.vocabulary()))
    print("Average paragraph length:", paragraph_index.avg_doc_length)
