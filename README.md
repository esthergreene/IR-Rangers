# IR-Rangers

Paper and paragraph retrieval over 34,308 arXiv papers: BM25 and TF-IDF
(Part 1), learned sparse (SPLADE) and dense (MiniLM) retrieval with rank
fusion (Part 2), and a web search interface.

COS 487/587 Introduction to Information Retrieval, Fall 2026, Course Project
Team: Esther Greene, Courtney Jackson, Nathaniel Serrano

Write-ups: [`paper/part1/paper.pdf`](paper/part1/paper.pdf),
[`paper/part2/paper.pdf`](paper/part2/paper.pdf)

## Results

Study queries (250). The neural models are fine-tuned on the Train queries
and rank papers by their best paragraph (MaxP). RRF fuses BM25, Dense and
SPLADE (k = 60). Significance tests are in the paper.

| Task       | Model  | nDCG  | nDCG@5 | MRR   | P@10  | R@100 |
|------------|--------|-------|--------|-------|-------|-------|
| Papers     | BM25   | 0.582 | 0.514  | 0.544 | 0.080 | 0.803 |
| Papers     | Dense  | 0.640 | 0.592  | 0.631 | 0.084 | 0.807 |
| Papers     | SPLADE | 0.671 | 0.631  | 0.658 | 0.090 | 0.825 |
| Papers     | RRF    | 0.701 | 0.655  | 0.689 | 0.095 | 0.853 |
| Paragraphs | BM25   | 0.474 | 0.416  | 0.516 | 0.107 | 0.592 |
| Paragraphs | Dense  | 0.447 | 0.377  | 0.477 | 0.104 | 0.607 |
| Paragraphs | SPLADE | 0.494 | 0.434  | 0.539 | 0.116 | 0.630 |
| Paragraphs | RRF    | 0.550 | 0.489  | 0.582 | 0.130 | 0.689 |

## Requirements

- Linux or macOS with `bash`, and conda (installed by `install.sh` if missing)
- An NVIDIA GPU for fine-tuning and building the neural indexes
- About 8 GB of free memory (each paragraph index takes 3.5–6.5 GB once loaded)
- The course dataset in `data/raw/` (`install.sh` links it on the CS servers):
  `JSON Files/`, `Train.json`, `Study.json`, `Test.json`, and the Train and
  Study paper and paragraph qrels

## Quick start

```bash
bash scripts/install.sh
```

```bash
bash scripts/run.sh --train
```

```bash
bash scripts/run.sh --interface
```

`install.sh` creates the `ir-rangers` conda environment (Python 3.11,
PyTorch, Node.js 22) and installs the frontend. `run.sh` builds the
collections and indexes, fine-tunes both neural models (`--train`), runs
BM25, SPLADE and Dense on the Study and Test queries, evaluates the Study
runs with rank fusion into `results/`, and copies the eight neural run files
to `Retrieval Results/`. Finished steps are skipped (`--force` rebuilds), and
output is logged to `logs/`.

Checkpoints are not in the repository, so a new machine needs `--train`.
Without it, `run.sh` uses `checkpoints/sparse/run2` and `checkpoints/dense/run1`
and takes about 30 minutes once the indexes exist. `--zeroshot` also
evaluates the models without fine-tuning.

`--interface` starts the search interface (BM25 and TF-IDF) at
http://127.0.0.1:5173; forward the port over SSH and turn **Sample data** off.

## Running steps individually

After `conda activate ir-rangers`, the Part 2 pipeline is:

```bash
python scripts/build_text_collections.py                                   # cleaned paragraphs
python scripts/run_retrieval.py --collection paragraph --queries train --model bm25
python scripts/build_training_data.py                                      # BM25 hard negatives
python -m src.training.train_splade --run-name run2
python -m src.training.train_dense --run-name run1
python -m src.index.sparse_indexer --model checkpoints/sparse/run2/best_model --index data/processed_neural/sparse_paragraph_index_ft.npz
python -m src.index.dense_indexer --model checkpoints/dense/run1/best_model --output-dir data/processed_neural/dense_run1
python scripts/run_retrieval.py --collection paper --queries study --model sparse --checkpoint checkpoints/sparse/run2/best_model --index data/processed_neural/sparse_paragraph_index_ft.npz
python -m src.eval.evaluation --queries study --models bm25 dense sparse rrf:bm25+dense+sparse
```

Search with the same encoder that built the index. Leave out `--model` and
`--checkpoint` for the zero-shot base models, and use `--run-name` to name
the run (e.g., `sparse_zeroshot`). Part 1 uses `build_collections.py`,
`python -m src.index.indexer`, and `run_retrieval.py --model bm25` or
`tfidf`; BM25's tuned parameters are in `config/bm25.yaml` (tuned with
`scripts/tune_bm25.py`). Preprocessing settings are in `config/`.

## Repository structure

```
config/              preprocessing and BM25 settings
src/preprocessing/   tokenizer and text cleaning
src/index/           inverted, SPLADE and FAISS indexes
src/model/           BM25, TF-IDF, SPLADE and dense searchers
src/training/        fine-tuning, training and validation data
src/eval/            evaluation and rank fusion (ranx)
src/common/          queries, run files, MaxP, paragraph loading
scripts/             install, run, collections, retrieval, tuning, plots
interface/           FastAPI backend and React frontend
annotation/          our Test query judgments (Parts 1 and 2)
paper/               LaTeX source and PDF of each part
```

## Contributions

### Part 2

Courtney:

- Built the SPLADE sparse encoder and index
- Built the sparse search model
- Wrote the SPLADE fine-tuning script (contrastive loss with FLOPS regularization)
- Wrote script to plot training and validation loss for both models

Esther:

- Built the Sentence-BERT (MiniLM) dense retriever
- Built the FAISS paragraph index, with checkpoint recovery
- Wrote the dense fine-tuning script (training configuration, early stopping)
- Ran zero-shot vs. fine-tuned dense evaluation

Nathaniel:

- Built the neural text preprocessing and cleaned paragraph collection
- Built paper ranking from paragraphs (MaxP)
- Built the training data: train/validation split, positives, BM25 hard negatives
- Built the shared validation set used to pick the best checkpoint
- Extended the codebase for the new models (retrieval and evaluation scripts,
  rank fusion, integrating both models into the shared pipeline, run script)

All together: Test query annotation and the paper.

### Part 1

- **Preprocessing, BM25 and tuning:** Nathaniel
- **Indexing:** Esther
- **TF-IDF and evaluation script:** Courtney (nDCG and nDCG@5: Esther)
- **Search interface:** frontend initially prompted by Nathaniel using Claude
  Opus 4.5, with manual edits by Courtney; backend by Esther and Courtney
- **Data annotation and paper:** all together