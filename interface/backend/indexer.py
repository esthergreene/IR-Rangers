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

if __name__ == "__main__":
    files = os.listdir(DATA_FOLDER)
    json_files = [file for file in files if file.endswith(".json")]

    print("Number of JSON files: ", len(json_files))

    first_file = json_files[0]
    file_path = os.path.join(DATA_FOLDER, first_file)
    paper = load_paper(file_path)

    print("Paper ID: ", paper["paper_id"])
    print("Title: ", paper["title"])
    print("Abstract: ", paper["abstract"])
    print("Number of paragraphs: ", len(paper["paragraphs"]))
    print("Processed title:", preprocess(paper["title"]))

    paper_text = paper["title"] + " " + paper["abstract"]

    for paragraph in paper["paragraphs"].values():
        paper_text += " " + paragraph

    paper_tokens = preprocess(paper_text)
    term_frequencies = Counter(paper_tokens)

    print("Total tokens: ", len(paper_tokens))
    print("Unique terms: ", len(term_frequencies))
    print("10 most common terms: ", term_frequencies.most_common(10))

    paper_index = {}
    document_lengths = {}
    document_frequency = {}

    for filename in json_files[:5]:
        file_path = os.path.join(DATA_FOLDER, filename)
        paper = load_paper(file_path)
        paper_text = paper["title"] + " " + paper["abstract"]

        for paragraph in paper["paragraphs"].values():
            paper_text += " " + paragraph

        paper_tokens = preprocess(paper_text)
        term_frequencies = Counter(paper_tokens)
        document_lengths[paper["paper_id"]] = len(paper_tokens)

        for term, frequency in term_frequencies.items():
            if term not in paper_index:
                paper_index[term] = {}

            paper_index[term][paper["paper_id"]] = frequency

    for term, documents in paper_index.items():
        document_frequency[term] = len(documents)

    print("Papers indexed: ", 5)
    print("Unique terms in index: ", len(paper_index))
    print("Index entry for diffusion: ", paper_index.get("diffusion"))
    print("Index entry for model:", paper_index.get("model"))
    print("Document lengths:", document_lengths)
    print("Document frequency for model:", document_frequency.get("model"))
    print("Document frequency for diffusion:", document_frequency.get("diffusion"))