"""
Print basic statistics about the paper collection.

Counts papers and paragraphs, summarizes paragraph lengths (in tokens after
preprocessing), and lists the most frequent tokens. Use it to spot noise such
as citation numbers, URLs, and LaTeX leftovers before building the index.

Usage, from the repository root:
    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --config config/preprocessing_nostem.yaml --limit 200
"""
import argparse
import json
import statistics
import sys
import time
import heapq
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.tokenizer import DEFAULT_CONFIG_PATH, Tokenizer  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "JSON Files"


def parse_args():
    parser = argparse.ArgumentParser(description="Print statistics about the paper collection.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="Folder of per-paper JSON files")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Tokenizer config file to use")
    parser.add_argument("--top", type=int, default=50,
                        help="How many of the most frequent tokens to list")
    parser.add_argument("--short", type=int, default=10,
                        help="Paragraphs with fewer tokens than this count as short")
    parser.add_argument("--long", type=int, default=1000,
                        help="Paragraphs with more tokens than this count as long")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N files (for quick tests)")
    return parser.parse_args()


def percentile(sorted_values, fraction):
    """Return the value at the given fraction (0 to 1) of a sorted list."""
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def main():
    args = parse_args()
    tokenizer = Tokenizer(config_path=args.config)

    files = sorted(args.data.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No .json files found in {args.data}")

    token_counts = Counter()
    paragraph_lengths = []
    short_examples = []
    longest = []  # min-heap of (length, paragraph_id, preview); holds the 10 longest so far
    unreadable_files = []
    num_papers = 0
    num_empty = 0
    num_odd_latex_math = 0

    start = time.perf_counter()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                paper = json.load(f)
        except json.JSONDecodeError:
            unreadable_files.append(path.name)
            continue

        num_papers += 1

        # Only paragraphs are counted. The abstract field repeats paragraph 1,
        # so counting it too would double those tokens.
        for paragraph_id, text in paper.get("paragraphs", {}).items():
            if text.count("$") % 2 == 1:
                num_odd_latex_math += 1
            tokens = tokenizer.tokenize(text)
            paragraph_lengths.append(len(tokens))
            token_counts.update(tokens)

            if not tokens:
                num_empty += 1
            if len(tokens) < args.short and len(short_examples) < 5:
                short_examples.append((paragraph_id, text[:100]))
            entry = (len(tokens), paragraph_id, text[:100])
            if len(longest) < 10:
                heapq.heappush(longest, entry)
            else:
                heapq.heappushpop(longest, entry)
    elapsed = time.perf_counter() - start

    if not paragraph_lengths:
        sys.exit("No paragraphs found.")

    lengths = sorted(paragraph_lengths)
    num_short = sum(1 for n in lengths if n < args.short)
    num_long = sum(1 for n in lengths if n > args.long)
    singletons = [token for token, count in token_counts.items() if count == 1]

    print(f"Config:     {args.config}")
    print(f"Papers:     {num_papers:,}  (processed in {elapsed:.1f}s)")
    if unreadable_files:
        print(f"Unreadable: {len(unreadable_files)} files, e.g. {unreadable_files[:5]}")
    print(f"Paragraphs: {len(lengths):,}  ({len(lengths) / num_papers:.1f} per paper)")

    print("\nParagraph length (tokens after preprocessing)")
    print(f"  min {lengths[0]}   median {statistics.median(lengths)}   "
          f"90th percentile {percentile(lengths, 0.9)}   max {lengths[-1]}")
    print(f"  empty: {num_empty:,}   shorter than {args.short} tokens: {num_short:,}")
    print(f"  odd LaTeX math: {num_odd_latex_math:,}")
    for paragraph_id, preview in short_examples:
        print(f"    {paragraph_id}: {preview!r}")
    print(f"  longer than {args.long} tokens: {num_long:,}")
    print("  Longest paragraphs:")
    for length, paragraph_id, preview in sorted(longest, reverse=True):
        print(f"    {paragraph_id} ({length:,} tokens): {preview!r}")
    print(f"\nTokens: {sum(lengths):,} total, {len(token_counts):,} unique, "
          f"{len(singletons):,} appear only once")
    print(f"  Examples of tokens that appear once: {singletons[:20]}")

    print(f"\nTop {args.top} tokens:")
    for rank, (token, count) in enumerate(token_counts.most_common(args.top), start=1):
        print(f"  {rank:>3}. {token:<20} {count:,}")


if __name__ == "__main__":
    main()