#!/usr/bin/env bash
# Run the COS 487 / 587 IR Rangers Project Part 1
#
# Experiments (Default):
#   1. Build the paper and paragraph collections
#   2. Build the inverted indices and the paper metadata for the interface
#   3. Run TF-IDF and BM25 on the Study and Test queries (8 run files)
#   4. Evaluate the study runs
#   5. Check the run files and copy them into "Retrieval Results/"
#
# Web app (--interface): starts the FastAPI backend and the frontend dev server.
#
# Usage, from anywhere:
#   bash scripts/run.sh                             # experiments only
#   bash scripts/run.sh --interface                 # web app only
#   bash scripts/run.sh --experiments --interface   # experiments, then the web app
#   bash scripts/run.sh --force                     # rebuild collections and indices
set -euo pipefail

ENV_NAME="ir-rangers"
HOST="127.0.0.1"
BACKEND_PORT=8765
FRONTEND_PORT=5173
BACKEND_STARTUP_SECONDS=120

PROCESSED="data/processed"
RUNS="runs"
RESULTS_DIR="Retrieval Results"
FRONTEND_DIR="interface/frontend"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- Options ----
RUN_EXPERIMENTS=false
INTERFACE=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --experiments)  RUN_EXPERIMENTS=true ;;
        --interface)    INTERFACE=true ;;
        --force)        FORCE=true ;;
        -h| --help)     usage; exit 0 ;;
        *)              echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

# If no --experiments or --interface flags, run experiments
if ! $RUN_EXPERIMENTS && ! $INTERFACE; then
    RUN_EXPERIMENTS=true
fi

# ---- Helpers ----
step() {
    echo
    echo "=== $* ($(date +%H:%M:%S))"
}

require() {
    # require <path> <hint>: stop with a clear message if a needed file is missing
    if [ ! -e "$1" ]; then
        echo "Missing $1. $2" >&2
        exit 1
    fi
}

# Conda's activation scripts reference unset variables, so pause set -u briefly
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# ---- Experiments ----
run_experiments() {
    step "Checking inputs"
    for path in "data/raw/JSON Files" data/raw/Study.json data/raw/Test.json \
                data/raw/Study_paper_qrel.tsv data/raw/Study_paragraph_qrel.tsv \
                config/bm25.yaml; do
        require "$path" "Run scripts/install.sh, or see the README for where the data goes."
    done
    echo "All inputs found."

    step "1/5 Building Collections"
    # build_collections.py renames both files into place at the very end, so
    # paragraphs.jsonl existing means the build finished.
    local rebuilt=false
    if $FORCE || [ ! -f "$PROCESSED/paragraphs.jsonl" ]; then
        python scripts/build_collections.py
        rebuilt=true
    else
        echo "Found $PROCESSED/paragraphs.jsonl; skipping (use --force to rebuild)."
    fi

    step "2/5 Building indexes and paper metadata"
    # The indexer writes the paragraph index last, so its presence means both exist
    # Everything here is rebuilt whenever the collections were
    if $FORCE || $rebuilt || [ ! -f "$PROCESSED/paragraph_index.json" ]; then
        python -m src.index.indexer
    else
        echo "Found $PROCESSED/paragraph_index.json; skipping (use --force to rebuild)."
    fi
    if $FORCE || $rebuilt || [ ! -f "$PROCESSED/paper_metadata.jsonl" ]; then
        python scripts/build_paper_metadata.py
    else
        echo "Found $PROCESSED/paper_metadata.jsonl; skipping (use --force to rebuild)."
    fi
    
    step "3/5 Running Retrieval"
    # BM25 reads its tuned k1 and b for each collection from config/bm25.yaml
    for query_set in study test; do
        for collection in paper paragraph; do
            for model in tfidf bm25; do
                echo "--- $model / $collection / $query_set"
                python scripts/run_retrieval.py --collection "$collection" \
                    --queries "$query_set" --model "$model"
            done
        done
    done

    step "4/5 Evaluating Study runs"
    mkdir -p results
    python -m src.eval.evaluation --queries study | tee results/study_evaluation.txt

    step "5/5 Checking run files and copying them to \"$RESULTS_DIR\""
    mkdir -p "$RESULTS_DIR"
    for query_set in study test; do
        for collection in paper paragraph; do
            for model in tfidf bm25; do
                # Ask trec.py for name, so team name lives in one place
                local name
                name=$(python -c "from src.common.trec import run_filename; \
                print(run_filename('$collection', '$model', '$query_set'))")
                local run_file="$RUNS/$name"
                require "$run_file" "Retrieval did not produce it."
                # Every line must have 6 tab-separated columns, with at most 100 results per query
                if ! awk -F'\t' 'NF != 6 { bad = 1 }
                                 { count[$1]++ }
                                 END { for (q in count) if (count[q] > 100) bad = 1; exit bad }' \
                        "$run_file"; then
                    echo "Format problem in $run_file" >&2
                    exit 1
                fi
                echo "OK $name ($(cut -f1 "$run_file" | sort -u | wc -l | tr -d ' ') queries)"
                cp "$run_file" "$RESULTS_DIR/"
            done
        done
    done
}

# --- Search Interface (web app) ----
interface() {
    step "Starting the search interface"
    for path in "$PROCESSED/paper_index.json" "$PROCESSED/paper_metadata.jsonl" \
                "$PROCESSED/preprocessing_config.yaml"; do
        require "$path" "Run the experiments first: bash scripts/run.sh"
    done
    require "$FRONTEND_DIR/node_modules" "Run bash scripts/install.sh first."

    python -m uvicorn interface.backend.main:app --host "$HOST" --port "$BACKEND_PORT" &
    local backend_pid=$!
    # Stop the backend whenever this script exits, including on Ctrl-C.
    trap "kill $backend_pid 2>/dev/null || true" EXIT

    echo "Waiting for the backend (it loads the paper index first)..."
    local ready=false
    for _ in $(seq 1 "$BACKEND_STARTUP_SECONDS"); do
        if python -c "import urllib.request; \
urllib.request.urlopen('http://$HOST:$BACKEND_PORT/api/health', timeout=1)" 2>/dev/null; then
            ready=true
            break
        fi
        if ! kill -0 "$backend_pid" 2>/dev/null; then
            echo "The backend stopped during startup; see the error above." >&2
            exit 1
        fi
        sleep 1
    done
    if ! $ready; then
        echo "The backend did not respond within $BACKEND_STARTUP_SECONDS seconds." >&2
        exit 1
    fi

    echo
    echo "Backend ready on http://$HOST:$BACKEND_PORT"
    echo "Open http://$HOST:$FRONTEND_PORT in a browser (forward port $FRONTEND_PORT"
    echo "if you are connected over SSH), and turn Sample data off. Press Ctrl-C to stop."
    echo
    (cd "$FRONTEND_DIR" && npm run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort)
}

# --- Main ----
if $RUN_EXPERIMENTS; then
    mkdir -p logs
    LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
    # Save experiment output to a log file along with the screen.
    run_experiments 2>&1 | tee -a "$LOG"
    echo "Experiments finished in $((SECONDS / 60)) min $((SECONDS % 60)) s. Log: $LOG"
fi

if $INTERFACE; then
    interface
fi