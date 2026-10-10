import math

def dcg(relevance_scores):
    total = 0.0

    for rank, relevance in enumerate(relevance_scores, start=1):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)
        total += gain / discount

    return total

def ndcg(relevance_scores):
    actual_dcg = dcg(relevance_scores)
    ideal_scores = sorted(relevance_scores, reverse = True)
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def ndcg_at_5(relevance_scores):
    actual_dcg = dcg(relevance_scores[:5])
    ideal_scores = sorted(relevance_scores, reverse=True)[:5]
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def load_qrels(file_path):
    qrels = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            query_id, _, document_id, relevance = line.split()

            if query_id not in qrels:
                qrels[query_id] = {}

            qrels[query_id][document_id] = int(relevance)

    return qrels

def evaluate_ndcg(retrieved_ids, query_qrels, k=None):
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    relevance_scores = [
        query_qrels.get(doc_id, 0)
        for doc_id in retrieved_ids
    ] 

    actual_dcg = dcg(relevance_scores)
    ideal_scores = sorted(query_qrels.values(), reverse=True)

    if k is not None:
        ideal_scores = ideal_scores[:k]

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def load_run(file_path):
    runs = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            query_id, _, document_id, rank, score, run_id = line.split()

            if query_id not in runs:
                runs[query_id] = []

            runs[query_id].append((int(rank), document_id))

    for query_id in runs:
        runs[query_id].sort(key=lambda result: result[0])
        runs[query_id] = [
            document_id for rank, document_id in runs[query_id]
        ]

    return runs

def evaluate_all_queries(runs, qrels):
    ndcg_scores = []
    ndcg_at_5_scores = []

    for query_id, query_qrels in qrels.items():
        retrieved_ids = runs.get(query_id, [])
        ndcg_score = evaluate_ndcg(retrieved_ids, query_qrels, k=100)
        ndcg_at_5_score = evaluate_ndcg(retrieved_ids, query_qrels, k=5)
        ndcg_scores.append(ndcg_score)
        ndcg_at_5_scores.append(ndcg_at_5_score)

    if not qrels:
        return 0.0, 0.0

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    average_ndcg_at_5 = sum(ndcg_at_5_scores) / len(ndcg_at_5_scores)

    return average_ndcg, average_ndcg_at_5