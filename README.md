# IR-Rangers
Introduction to Information Retrieval, Fall 2026, Course Project

# Search Interface
- Frontend - Prompted by Nathaniel using Claude Opus 4.5, with manual edits by Courtney
- Backend - Esther and Nathaniel

# Models and Processing
- Processing - All together
- Indexing - Esther with assistance from group memebers
- TF-IDF - Courtney
- BM25 - Nathaniel

# Evaluation 
- nDCG - Esther
- nDCG@5 - Esther
- MRR - Nathaniel
- Precision@5 - Courtney
- Percision@10 - Courtney

Basic Script pre-data in file DUE October 2nd

# Data Annotation
- Worked on all together 

# Retrieval Results Files

# Paper
All together

## RUNNING THE SEARCH ENGINE

Run these commands from the root of the `IR-Rangers` repository.

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the search data

The project requires the original paper JSON files in `data/raw/JSON Files/`. These data files are not included in GitHub.

Generate the processed collections:

```bash
python scripts/build_collections.py
```

Build the paper and paragraph search indexes:

```bash
python -m src.index.indexer
```

Generate the paper metadata used by the frontend:

```bash
python scripts/build_paper_metadata.py
```

These steps create the processed data files required by the backend.

### 3. Start the FastAPI backend

From the repository root:

```bash
python -m uvicorn interface.backend.main:app --host 127.0.0.1 --port 8765
```

The backend will be available at `http://127.0.0.1:8765`.

### 4. Start the React frontend

Open a second terminal and run:

```bash
cd interface/frontend
npm install
npm run dev -- --host 127.0.0.1
```

The frontend requires Node.js 22 or a compatible newer version.

Open `http://127.0.0.1:5173` in your browser. If using VS Code Remote SSH, forward port 5173.

### 5. Search the collection

Turn **Sample data** off in the frontend menu or Settings page to search the real collection. Changing the model or collection in Settings also turns sample data off and searches the backend using the selected BM25/TF-IDF model and paper/paragraph collection. Restart any backend process that was already running before these settings were added.

Search results are shown 10 ranked research papers at a time, with page controls to browse additional matches.
