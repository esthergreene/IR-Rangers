import json
import os

DATA_FOLDER = os.path.expanduser("~/datasets/IR26/JSON Files")

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