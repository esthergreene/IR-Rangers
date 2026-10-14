# IR-Rangers

Paper and paragraph retrieval over 34,308 arXiv papers with TF-IDF and BM25,
plus a web search interface.

COS 487/587 Introduction to Information Retrieval, Fall 2026, Course Project Part 1
Team: Esther Greene, Courtney Jackson, Nathaniel Serrano

The full write-up is in [`paper/paper.pdf`](paper/paper.pdf).

## Results

Study queries (250), BM25 with tuned parameters. BM25 is significantly better
than TF-IDF on every metric for both tasks (paired t-test, p < 0.05).

| Task       | Model  | nDCG  | nDCG@5 | MRR   | P@5   | P@10  |
|------------|--------|-------|--------|-------|-------|-------|
| Papers     | TF-IDF | 0.397 | 0.313  | 0.332 | 0.094 | 0.058 |
| Papers     | BM25   | 0.582 | 0.514  | 0.544 | 0.137 | 0.080 |
| Paragraphs | TF-IDF | 0.301 | 0.229  | 0.298 | 0.103 | 0.067 |
| Paragraphs | BM25   | 0.474 | 0.416  | 0.516 | 0.174 | 0.107 |

Tuned BM25 parameters (`config/bm25.yaml`): papers k1 = 4.0, b = 0.75;
paragraphs k1 = 0.4, b = 0.9.

## Requirements

- Linux or macOS with `bash`
- conda (the install script installs Miniforge if none is found)
- Python 3.8 and Node.js 22+ (both installed by the install script)
- About 6 GB of free memory to search the paragraph collection (its index
  takes 5.5 GB once loaded)
- The course dataset (not included in this repository; see [Data](#data))

## Quick start

From the repository root:

```bash
bash scripts/install.sh
```

```bash
bash scripts/run.sh
```

```bash
bash scripts/run.sh --interface
```

1. `install.sh` creates the `ir-rangers` conda environment, installs the
   packages in `requirements.txt`, downloads the NLTK stopwords, installs
   Node.js and the frontend packages, and links `data/raw` to the shared
   dataset on the CS servers if it is available. It is safe to run again.
2. `run.sh` runs all experiments: it builds the collections and indexes,
   runs both models on the Study and Test queries, evaluates the Study runs,
   checks the eight run files and copies them to `Retrieval Results/`.
   Steps whose outputs already exist are skipped; add `--force` to rebuild
   everything. Output is also saved to `logs/`.
3. `run.sh --interface` starts the backend and the frontend. Open
   http://127.0.0.1:5173 (forward port 5173 if you are connected over SSH)
   and turn **Sample data** off to search the real collection.

## Data

Put the dataset in `data/raw/` (`install.sh` links it automatically on the
CS servers):

```
data/raw/
├── JSON Files/                 one JSON file per paper
├── Train.json  Study.json  Test.json
├── Train_paper_qrel.tsv        Train_paragraph_qrel.tsv
└── Study_paper_qrel.tsv        Study_paragraph_qrel.tsv
```

The Train files are needed only for tuning BM25.

## Outputs

| Path | Contents |
|---|---|
| `data/processed/` | Processed collections (`papers.jsonl`, `paragraphs.jsonl`), indexes, paper metadata, and a copy of the preprocessing config |
| `runs/` | TREC run files, top 100 results per query |
| `annotation/IRRangers_qrel_part1_test.tsv` | Our judgments of BM25's top-ranked paragraph for each Test query |
| `logs/` | Logs of `run.sh` |

## Running steps individually

Activate the environment first with `conda activate ir-rangers`.

```bash
python scripts/build_collections.py
```

```bash
python -m src.index.indexer
```

```bash
python scripts/build_paper_metadata.py
```

```bash
python scripts/run_retrieval.py --collection paper --queries study --model bm25
```

```bash
python -m src.eval.evaluation --queries study
```

`run_retrieval.py` reads BM25's parameters from `config/bm25.yaml`; pass
`--k1` and `--b` to override them, or `--limit-queries 20` for a quick test.
`build_collections.py --limit 200 --output data/processed_sample` builds a
small sample collection for testing.

### Preprocessing

Preprocessing settings are read from `config/preprocessing.yaml`: inline-math,
LaTeX command, URL and citation removal, tokenization, stopwords and Snowball
stemming. `preprocessing_nostem.yaml` and `preprocessing_baseline.yaml` are
the variants compared in the paper. Pass one with `--config`:

```bash
python scripts/corpus_stats.py --config config/preprocessing_baseline.yaml
```

### Tuning BM25

The paper's tuning used all 2,000 Train queries (1,600 for tuning, 400 held
out) and these grids:

```bash
python scripts/tune_bm25.py --collection paper --num-queries 2000 --k1 0.9 1.2 1.8 2.4 3 4 5 6 8 --b 0.3 0.5 0.6 0.75 0.9 1.0 --output results/bm25_tuning_paper_full.csv
```

```bash
python scripts/tune_bm25.py --collection paragraph --num-queries 2000 --k1 0.2 0.4 0.6 0.8 1.0 1.2 1.5 1.8 --b 0.3 0.5 0.6 0.75 0.9 1.0 --output results/bm25_tuning_paragraph_full.csv
```

```bash
python scripts/plot_tuning_heatmap.py results/bm25_tuning_paper_full.csv
```

## Search interface

The interface has a React frontend (`interface/frontend`) and a FastAPI
backend (`interface/backend/main.py`). To start them separately:

```bash
python -m uvicorn interface.backend.main:app --host 127.0.0.1 --port 8765
```

```bash
cd interface/frontend && npm run dev -- --host 127.0.0.1
```

The backend loads the paper index and BM25 at startup; the paragraph index
and TF-IDF load on first use. In the interface you can:

- choose the model (BM25 or TF-IDF) and collection (papers or paragraphs) in Settings
- see the query's terms and toggle their highlighting in result excerpts
- get title suggestions as you type
- preview documents, and bookmark them to named collections
- review and clear your search history

## Repository structure

```
config/              preprocessing and BM25 settings
src/preprocessing/   tokenizer
src/index/           inverted index
src/model/           BM25 and TF-IDF
src/eval/            evaluation with ranx
src/common/          query loading, TREC run files, BM25 config
scripts/             install, run, collection building, retrieval, tuning, plots
interface/           FastAPI backend and React frontend
annotation/          our Test query judgments
paper/               LaTeX source, figures and PDF of the paper
```

## Contributions

- **Preprocessing:** Nathaniel
- **Indexing:** Esther
- **TF-IDF:** Courtney
- **BM25 and tuning:** Nathaniel
- **Evaluation Script:** Courtney
   - nDCG and nDCG@5: Esther
- **Search interface:** frontend initially prompted by Nathaniel using Claude Opus 4.5, with manual edits by Courtney; backend by Esther and Courtney
- **Data annotation:** all together
- **Paper:** all together