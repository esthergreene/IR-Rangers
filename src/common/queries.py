"""
Loading query files (Train.json, Study.json, Test.json).

Each file is a JSON list of objects with "query_id", "query", and "label"
(Easy, Medium, or Hard). Query IDs are converted to strings so they match the
IDs in the QREL files and TREC run files.

Usage:
    from src.common.queries import load_queries
    queries = load_queries("data/raw/Study.json")
"""
import json
from dataclasses import dataclass

REQUIRED_FIELDS = ("query_id", "query")

# Kept frozen to ensure immutability of query objects.
@dataclass(frozen=True)
class Query:
    """One query from a query file."""

    query_id: str
    text: str
    label: str = None # Easy, Medium, or Hard; None if the file has no labels

def load_queries(path):
    """
    Load a query file.

    Args:
        path (str or Path): Path to a query JSON file.
    
    Returns:
        list of Query: Queries in file order.
    
    Raises:
        ValueError: If the file is not a JSON list, a query is missing a
        required field, or a query ID appears twice.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of queries.")
    
    queries = []
    seen_ids = set()
    for position, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{path}: query at position {position} is missing {missing}.")
        query_id = str(record["query_id"])
        if query_id in seen_ids:
            raise ValueError(f"{path}: query_id {quer_id} appears more than once.")
        seen_ids.add(query_id)
        
        queries.append(Query(
            query_id=query_id,
            text=record["query"],
            label=record.get("label")
        ))
    
    return queries