"""
Ranking papers from paragraph scores (MaxP).

The neural models only index paragraphs, so paper rankings are built from 
paragraph rankings: each paper is scored by its best-scoring paragraph.
BM25 and TF-IDF still use their own paper index and do not need this.

Retrieve paragraphs deeper than the number of papers needed (e.g., 2,000 
paragraphs for 100 papers), since one paper often has several paragraphs 
near the top of the ranking.

Usage:
    from src.common.maxp import load_paper_of, maxp

    paper_of = load_paper_of("data/processed_neural/paragraphs.jsonl)
    paragraphs = model.search(query_text, k=2000)
    papers = maxp(paragraphs, paper_of, k=100)
"""
import json

def load_paper_of(paragraphs_path):
    """
    Map each paragraph to its paper, from a cleaned-text paragraphs.jsonl

    Args:
        paragraphs_path (str or Path): Path to paragraphs.jsonl
    
    Returns:
        dict: paragraph_id -> paper_id
    """
    paper_of = {}
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            paper_of[record["doc_id"]] = record["paper_id"]
    return paper_of

def maxp(paragraph_results, paper_of, k=100):
    """
    Turn a ranked paragraph list into a ranked paper list.

    Each paper is scored by its best paragraph. Because paragraph_results is
    sorted by score, a paper's first appearance is its best paragraph, so 
    keeping first appearances gives the papers in MaxP order.

    Args: 
        paragraph_results (list): (paragraph_id, score) pairs, best first
        paper_of (dict): paragraph_id -> paper_id, from load_paper_of()
        k (int): Number of papers to return

    Returns:
        list: (paper_id, score) pairs, best first. Fewer than k if the paragraphs 
            came from fewer than k papers; retrieve more paragraphs if that happens
    """
    papers = []
    seen = set()
    for paragraph_id, score in paragraph_results:
        paper_id = paper_of[paragraph_id]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        papers.append((paper_id, score))
        if len(papers) == k:
            break
    return papers