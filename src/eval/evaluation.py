from ranx import Qrels, Run, evaluate, compare
from src.common.queries import load_queries
import argparse
import sys


def evaluate_metrics_at_2(qrels, run):
    qrels.set_relevance_level(rel_lvl=2)
    metrics = ["nDCG","nDCG@5", "MRR", "precision@5", "precision@10"]
    eval_res = evaluate(qrels, run, metrics, make_comparable=True)
    print(eval_res)

def evaluate_metrics_at_1(qrels, run):
    qrels.set_relevance_level(rel_lvl=1)
    metrics = ["MRR", "precision@5"]
    eval_res = evaluate(qrels, run, metrics, make_comparable=True)
    print(eval_res)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval outputs against qrels.")
    parser.add_argument("--collection", choices=["train", "test", "study"], required=True, help="Collection to use for evaluation")

    return parser.parse_args()

def main():
    args = parse_args()
    if parse_args == "test":
        run_bm25 = Run.from_file("/runs/test_bm25_qrel.tsv", kind="trec")
        run_tfidf = Run.from_file("/runs/test_tfidf_qrel.tsv", kind="trec")
        qrels_paper = Qrels.from_file("/data/raw/Test_paper_qrel.tsv", kind="trec")
        qrels_paragraph = Qrels.from_file("/data/raw/Test_paragraph_qrel.tsv", kind="trec")
    elif parse_args == "train":
        run_bm25 = Run.from_file("/runs/test_bm25_qrel.tsv", kind="trec")
        run_tfidf = Run.from_file("/runs/test_tfidf_qrel.tsv", kind="trec")
        qrels_paper = Qrels.from_file("/data/raw/Test_paper_qrel.tsv", kind="trec")
        qrels_paragraph = Qrels.from_file("/data/raw/Test_paragraph_qrel.tsv", kind="trec")
    elif parse_args == "study":
        run_bm25 = Run.from_file("/runs/study_bm25_qrel.tsv", kind="trec")
        run_tfidf = Run.from_file("/runs/study_tfidf_qrel.tsv", kind="trec")
        qrels_paper = Qrels.from_file("/data/raw/Test_paper_qrel.tsv", kind="trec")
        qrels_paragraph = Qrels.from_file("/data/raw/Test_paragraph_qrel.tsv", kind="trec")
    else:
        sys.exit("--collection must me either train, test, or study")
    
    runs = [run_bm25, run_tfidf]
    qrels = [qrels_paper, qrels_paragraph]

    metrics_at_2 = evaluate_metrics_at_2(qrels, runs)
    metrics_at_1 = evaluate_metrics_at_1(qrels, runs)


if __name__ == "__main__":
    main()

# retrievel outputs tsv file (take in a tsv file, found in runs/)
# Needs to iterate therough lines and give the metrics for everything
# qid, Q0, docid rank score runid
# iterate throuhg each line and evalute (whole file) on each metric
# when I run the eval script from command line, add argument where it will be --collection flag, 
# and then it will know which qrel to use.
# import the load file from queries.py
# qrel is the source of truth, the actually grade of the searhc results against the queies 
# in the proper order ex doc 12 is ranked 1 for this query if that is the case in the qrel treat that as 
# fact, point of qrel is something to comapre againt. queies is the test.json. this script will compares the qrel 
# vs our run. Flag to pass what query the user wants to evaluate (study, test, or train)
# be able to change the relevance setting, for mrr precision need consider 1 and 2 
# statistic signifacane testing between the two models (idf vs bm25)
# print a table with the metrics