import json
import os
from collections import Counter
from src.preprocessing.tokenizer import tokenize_text

DATA_FOLDER = os.path.expanduser("~/datasets/IR26/JSON Files")

def preprocess(text):
    return tokenize_text(text)

def load_paper(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        paper = json.load(file)

    return paper

def get_paper_text(paper):
    paper_text = paper["title"] + " " + paper["abstract"]

    for paragraph in paper["paragraphs"].values():
        paper_text += " " + paragraph
    
    return paper_text

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

def load_index(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return CorpusIndex.from_dict(data)
        
def build_paper_index(json_files):
    paper_index = CorpusIndex()

    for filename in json_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        paper = load_paper(file_path)
        paper_text = get_paper_text(paper)
        paper_tokens = preprocess(paper_text)
        term_frequencies = Counter(paper_tokens)
        doc_number = paper_index.num_docs
        paper_index._doc_ids.append(paper["paper_id"])
        paper_index._doc_numbers[paper["paper_id"]] = doc_number
        paper_index.doc_lengths.append(len(paper_tokens))

        for term, frequency in term_frequencies.items():
            if term not in paper_index._postings:
                paper_index._postings[term] = ([], [])

            paper_index._postings[term][0].append(doc_number)
            paper_index._postings[term][1].append(frequency)

        paper_index.num_docs += 1
        
    if paper_index.num_docs > 0:
        paper_index.avg_doc_length = sum(paper_index.doc_lengths) / paper_index.num_docs

    return paper_index

def build_paragraph_index(json_files):
    paragraph_index = CorpusIndex()

    for filename in json_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        paper = load_paper(file_path)

        for paragraph_id, paragraph_text in paper["paragraphs"].items():
            paragraph_tokens = preprocess(paragraph_text)
            term_frequencies = Counter(paragraph_tokens)
            doc_number = paragraph_index.num_docs
            paragraph_index._doc_ids.append(paragraph_id)
            paragraph_index._doc_numbers[paragraph_id] = doc_number
            paragraph_index.doc_lengths.append(len(paragraph_tokens))

            for term, frequency in term_frequencies.items():
                if term not in paragraph_index._postings:
                    paragraph_index._postings[term] = ([], [])

                paragraph_index._postings[term][0].append(doc_number)
                paragraph_index._postings[term][1].append(frequency)

            paragraph_index.num_docs += 1

    if paragraph_index.num_docs > 0:
        paragraph_index.avg_doc_length = sum(paragraph_index.doc_lengths) / paragraph_index.num_docs

    return paragraph_index

if __name__ == "__main__":
    files = os.listdir(DATA_FOLDER)
    json_files = sorted([file for file in files if file.endswith(".json")])

    print("Number of JSON files:", len(json_files))

    paper_index = build_paper_index(json_files)
    print("Papers indexed:", paper_index.num_docs)
    print("Unique terms in paper index:", len(paper_index.vocabulary()))
    print("Average paper length:", paper_index.avg_doc_length)

    save_json(paper_index.to_dict(), "paper_index.json")

    del paper_index

    paragraph_index = build_paragraph_index(json_files)
    print("Paragraphs indexed:", paragraph_index.num_docs)
    print("Unique terms in paragraph index:", len(paragraph_index.vocabulary()))
    print("Average paragraph length:", paragraph_index.avg_doc_length)

    save_json(paragraph_index.to_dict(), "paragraph_index.json")