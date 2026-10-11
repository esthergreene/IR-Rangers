"""
Tuned BM25 parameters for each collection, read from config/bm25.yaml.

Usage:
    from src.common.bm25_config import load_bm25_params
    k1, b = load_bm25_params("paper")
"""
from pathlib import Path

import yaml

# This file lives at src/common/bm25_config.py, so the repo root is two folders up.
REPO_ROOT = Path(__file__).resolve().parents[2]
BM25_CONFIG_PATH = REPO_ROOT / "config" / "bm25.yaml"


def load_bm25_params(collection, config_path=BM25_CONFIG_PATH):
    """
    Return the tuned (k1, b) for one collection.

    Args:
        collection (str): "paper" or "paragraph".
        config_path (str or Path): The BM25 config file.

    Returns:
        tuple of (float, float): k1 and b.

    Raises:
        ValueError: If the config has no k1/b for the collection.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    try:
        params = config[collection]
        return float(params["k1"]), float(params["b"])
    except (KeyError, TypeError):
        raise ValueError(f"{config_path} has no k1/b for the '{collection}' collection.") from None