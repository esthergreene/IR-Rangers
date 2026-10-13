"""
Plot a BM25 tuning grid (CSV from tune_bm25.py) as a heatmap for the paper.

Rows are k1, columns are b, and each cell shows the tuning-query score.
The best cell is outlined and its label is bold. The figure is sized for
one ACM column and saved as a vector PDF.

Usage, from the repository root:
    python scripts/plot_tuning_heatmap.py results/bm25_tuning_paper_full.csv
    python scripts/plot_tuning_heatmap.py results/bm25_tuning_paragraph_full.csv \
        --output paper/figures/bm25_tuning_paragraph.pdf
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no display needed on the server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Single-hue sequential ramp, light (low scores) to dark (high scores).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#1f1f1f"          # cell labels on light cells
INK_ON_DARK = "#ffffff"  # cell labels on dark cells
MUTED = "#6b6b6b"        # axis labels and ticks
CELL_GAP = "#ffffff"     # thin lines separating cells
HIGHLIGHT = "#eb6834"    # best-cell outline; stands out against the blue scale and in grayscale

# How metric names from the CSV (ranx's names) are written in the paper.
DISPLAY_NAMES = {"ndcg": "nDCG", "ndcg@5": "nDCG@5", "mrr": "MRR"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a BM25 tuning grid as a heatmap.")
    parser.add_argument("csv", type=Path, help="Grid CSV written by tune_bm25.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: paper/figures/<csv name>.pdf)")
    return parser.parse_args()


def read_grid(path):
    """
    Read a tuning CSV into a k1-by-b grid.

    Returns:
        (metric name, sorted k1 values, sorted b values, 2-D array of scores).
        Cells missing from the CSV are NaN.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metric = next(column for column in rows[0] if column not in ("k1", "b", "seconds"))

    k1_values = sorted({float(row["k1"]) for row in rows})
    b_values = sorted({float(row["b"]) for row in rows})
    grid = np.full((len(k1_values), len(b_values)), np.nan)
    for row in rows:
        grid[k1_values.index(float(row["k1"])), b_values.index(float(row["b"]))] = float(row[metric])
    return metric, k1_values, b_values, grid


def main():
    args = parse_args()
    output = args.output or REPO_ROOT / "paper" / "figures" / f"{args.csv.stem}.pdf"
    metric, k1_values, b_values, grid = read_grid(args.csv)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))  # about one ACM column wide
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    image = ax.imshow(grid, cmap=cmap, origin="lower", aspect="auto")  # k1 increases upward

    # Axes: labels only, no frame or tick marks, so the cells carry the figure.
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels([f"{b:g}" for b in b_values])
    ax.set_yticks(range(len(k1_values)))
    ax.set_yticklabels([f"{k1:g}" for k1 in k1_values])
    ax.set_xlabel("$b$", color=MUTED, fontsize=8)
    ax.set_ylabel("$k_1$", color=MUTED, fontsize=8)
    ax.tick_params(length=0, colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Thin gaps between cells.
    ax.set_xticks(np.arange(-0.5, len(b_values)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(k1_values)), minor=True)
    ax.grid(which="minor", color=CELL_GAP, linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # The best cell gets a bold label and an outline.
    best_i, best_j = np.unravel_index(np.nanargmax(grid), grid.shape)

    # Score in each cell, in whichever ink contrasts with the cell color.
    for i in range(len(k1_values)):
        for j in range(len(b_values)):
            value = grid[i, j]
            if np.isnan(value):
                continue
            text_color = INK_ON_DARK if image.norm(value) > 0.5 else INK
            weight = "bold" if (i, j) == (best_i, best_j) else "normal"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    fontsize=5.5, color=text_color, fontweight=weight)

    # Inset slightly and drawn above the gap lines, so the outline stays fully visible.
    ax.add_patch(Rectangle((best_j - 0.45, best_i - 0.45), 0.9, 0.9,
                           fill=False, edgecolor=HIGHLIGHT, linewidth=2, zorder=3))

    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label(f"{DISPLAY_NAMES.get(metric, metric)} (tuning queries)",
                       color=MUTED, fontsize=7)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, colors=MUTED, labelsize=6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()