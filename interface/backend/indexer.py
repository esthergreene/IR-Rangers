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

def calculate_document_frequency(index):
    document_frequency = {}

    for term, documents in index.items():
        document_frequency[term] = len(documents)

    return document_frequency

def build_paper_index(json_files):
    paper_index = {}
    document_lengths = {}

    for filename in json_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        paper = load_paper(file_path)
        paper_text = get_paper_text(paper)
        paper_tokens = preprocess(paper_text)
        term_frequencies = Counter(paper_tokens)
        document_lengths[paper["paper_id"]] = len(paper_tokens)

        for term, frequency in term_frequencies.items():
            if term not in paper_index:
                paper_index[term] = {}

            paper_index[term][paper["paper_id"]] = frequency
        
    return paper_index, document_lengths

def build_paragraph_index(json_files):
    paragraph_index = {}
    paragraph_lengths = {}

    for filename in json_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        paper = load_paper(file_path)

        for paragraph_id, paragraph_text in paper["paragraphs"].items():
            paragraph_tokens = preprocess(paragraph_text)
            term_frequencies = Counter(paragraph_tokens)
            paragraph_lengths[paragraph_id] = len(paragraph_tokens)

            for term, frequency in term_frequencies.items():
                if term not in paragraph_index:
                    paragraph_index[term] = {}

                paragraph_index[term][paragraph_id] = frequency

    return paragraph_index, paragraph_lengths

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

    paper_index, document_lengths = build_paper_index(json_files[:5])
    average_document_length = sum(document_lengths.values()) / len(document_lengths)
    document_frequency = calculate_document_frequency(paper_index)

    print("Papers indexed: ", 5)
    print("Unique terms in index: ", len(paper_index))
    print("Index entry for diffusion: ", paper_index.get("diffusion"))
    print("Index entry for model:", paper_index.get("model"))
    print("Document lengths:", document_lengths)
    print("Document frequency for model:", document_frequency.get("model"))
    print("Document frequency for diffusion:", document_frequency.get("diffusion"))
    print("Average document length:", average_document_length)
    print("\nFirst 3 paragraphs: ")

    for paragraph_id, paragraph_text in list(paper["paragraphs"].items())[:3]:
        print("Paragraph ID: ", paragraph_id)
        print("Processed length: ", len(preprocess(paragraph_text)))

    paragraph_index, paragraph_lengths = build_paragraph_index(json_files[:5])
    average_paragraph_length = sum(paragraph_lengths.values()) / len(paragraph_lengths)
    paragraph_frequency = calculate_document_frequency(paragraph_index)

    print("Paragraphs indexed:", len(paragraph_lengths))
    print("Unique terms in paragraph index:", len(paragraph_index))
    print("Paragraph index entry for model:", paragraph_index.get("model"))
    print("Paragraph frequency for model:", paragraph_frequency.get("model"))
    print("Average paragraph length:", average_paragraph_length)