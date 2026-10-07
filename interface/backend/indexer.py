import json
import os
import re
from nltk.corpus import stopwords
from collections import Counter

DATA_FOLDER = os.path.expanduser("~/datasets/IR26/JSON Files")
STOPWORDS = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [token for token in tokens if token not in STOPWORDS]

    return tokens

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

    print("Number of JSON files: ", len(json_files))

    first_file = json_files[0]
    file_path = os.path.join(DATA_FOLDER, first_file)
    paper = load_paper(file_path)

    print("Paper ID: ", paper["paper_id"])
    print("Title: ", paper["title"])
    print("Abstract: ", paper["abstract"])
    print("Number of paragraphs: ", len(paper["paragraphs"]))
    print("Processed title:", preprocess(paper["title"]))

    paper_text = get_paper_text(paper)
    paper_tokens = preprocess(paper_text)
    term_frequencies = Counter(paper_tokens)

    print("Total tokens: ", len(paper_tokens))
    print("Unique terms: ", len(term_frequencies))
    print("10 most common terms: ", term_frequencies.most_common(10))

    paper_index = build_paper_index(json_files[:5])

    print("Papers indexed: ", paper_index.num_docs)
    print("Unique terms in index: ", len(paper_index.vocabulary()))
    print("Index entry for diffusion: ", paper_index.postings("diffusion"))
    print("Index entry for model:", paper_index.postings("model"))
    print("Document lengths:", paper_index.doc_lengths)
    print("Document frequency for model:", paper_index.df("model"))
    print("Document frequency for diffusion:", paper_index.df("diffusion"))
    print("Average document length:", paper_index.avg_doc_length)
    print("Document ID for number 0:", paper_index.doc_id(0))
    print("Document number for ID 2408.00001:", paper_index.doc_number("2408.00001"))
    print("\nFirst 3 paragraphs: ")

    for paragraph_id, paragraph_text in list(paper["paragraphs"].items())[:3]:
        print("Paragraph ID: ", paragraph_id)
        print("Processed length: ", len(preprocess(paragraph_text)))

    paragraph_index = build_paragraph_index(json_files[:5])
    save_json(paper_index.to_dict(), "test_paper_index.json")
    save_json(paragraph_index.to_dict(), "test_paragraph_index.json")

    print("Paragraphs indexed:", paragraph_index.num_docs)
    print("Unique terms in paragraph index:", len(paragraph_index.vocabulary()))
    print("Paragraph frequency for model:", paragraph_index.df("model"))
    print("Average paragraph length:", paragraph_index.avg_doc_length)

    loaded_paper_index = load_index("test_paper_index.json")

    print("\nLoaded index test:")
    print("Loaded papers:", loaded_paper_index.num_docs)
    print("Loaded model postings:", loaded_paper_index.postings("model"))
    print("Loaded document ID for number 0:", loaded_paper_index.doc_id(0))
    print("Loaded document number for ID 2408.00001:", loaded_paper_index.doc_number("2408.00001"))

    loaded_paragraph_index = load_index("test_paragraph_index.json")

    print("\nLoaded paragraph index test:")
    print("Loaded paragraphs:", loaded_paragraph_index.num_docs)
    print("Loaded model frequency:", loaded_paragraph_index.df("model"))
    print("Loaded document ID for number 0:", loaded_paragraph_index.doc_id(0))