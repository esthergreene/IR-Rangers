#!/usr/bin/env bash
# Run the COS 487 / 587 IR Rangers Project Part 2
#
# Experiments (default):
#   1. Build the collections, the BM25 indexes, and the cleaned paragraph collection
#   2. Fine-tune SPLADE and the dense model (only with --train)
#   3. Build the SPLADE and dense paragraph indexes
#   4. Run BM25, dense, and SPLADE on the Study queries, and dense and SPLADE on Test
#   5. Evaluate the Study runs, including rank fusion
#   6. Check the 8 neural run files and copy them into "Retrieval Results/"
#
# Web app (--interface): starts the FastAPI backend and the frontend dev server.
#
# Steps 2-4 need a GPU. Without --train, the fine-tuned checkpoints must already be in checkpoints/.
#
# Usage, from anywhere:
#   bash scripts/run.sh                             # experiments only
#   bash scripts/run.sh --train                     # fine-tune both models first
#   bash scripts/run.sh --zeroshot                  # also run and evaluate the zero-shot models
#   bash scripts/run.sh --interface                 # web app only
#   bash scripts/run.sh --experiments --interface   # experiments, then the web app
#   bash scripts/run.sh --force                     # rebuild collections and indexes
set -euo pipefail

ENV_NAME="ir-rangers"
HOST="127.0.0.1"
BACKEND_PORT=8765
FRONTEND_PORT=5173
BACKEND_STARTUP_SECONDS=120

PROCESSED="data/processed"
NEURAL="data/processed_neural"
RUNS="runs"
RESULTS_DIR="Retrieval Results"
FRONTEND_DIR="interface/frontend"

# Fine-tuned models used in the paper, and their paragraph indexes
SPARSE_RUN="run2"
DENSE_RUN="run1"
SPARSE_CHECKPOINT="checkpoints/sparse/$SPARSE_RUN/best_model"
DENSE_CHECKPOINT="checkpoints/dense/$DENSE_RUN/best_model"
SPARSE_INDEX="$NEURAL/sparse_paragraph_index_ft.npz"
DENSE_INDEX="$NEURAL/dense_$DENSE_RUN/paragraph_dense.faiss"
# Zero-shot indexes (--zeroshot), built with each indexer's default base model
SPARSE_ZEROSHOT_INDEX="$NEURAL/sparse_paragraph_index.npz"
DENSE_ZEROSHOT_INDEX="$NEURAL/dense_zeroshot/paragraph_dense.faiss"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- Options ----
RUN_EXPERIMENTS=false
INTERFACE=false
FORCE=false
TRAIN=false
ZEROSHOT=false
for arg in "$@"; do
    case "$arg" in
        --experiments)  RUN_EXPERIMENTS=true ;;
        --interface)    INTERFACE=true ;;
        --force)        FORCE=true ;;
        --train)        TRAIN=true; RUN_EXPERIMENTS=true ;;
        --zeroshot)     ZEROSHOT=true; RUN_EXPERIMENTS=true ;;
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

run_path() {
    # run_path <collection> <model> <query set>: ask trec.py, so the team name lives in one place
    echo "$RUNS/$(python -c "from src.common.trec import run_filename; \
print(run_filename('$1', '$2', '$3'))")"
}

train_model() {
    # train_model <sparse|dense> <run name>: never overwrite a checkpoint folder
    local module="src.training.train_splade"
    [ "$1" = "dense" ] && module="src.training.train_dense"
    if [ -e "checkpoints/$1/$2" ]; then
        echo "checkpoints/$1/$2 already exists; move it aside to retrain." >&2
        exit 1
    fi
    python -m "$module" --run-name "$2"
}

build_sparse_index() {
    # build_sparse_index <index file> <rebuild true/false> [encoder]
    if $2 || [ ! -f "$1" ]; then
        python -m src.index.sparse_indexer --index "$1" --force ${3:+--model "$3"}
    else
        echo "Found $1; skipping (use --force to rebuild)."
    fi
}

build_dense_index() {
    # build_dense_index <index file> <rebuild true/false> [encoder]
    # The indexer resumes from a partial checkpoint, so clear it for a full rebuild
    local dir
    dir=$(dirname "$1")
    if $2 || [ ! -f "$1" ]; then
        if $2; then
            rm -f "$dir"/paragraph_dense_checkpoint*
        fi
        python -m src.index.dense_indexer --output-dir "$dir" ${3:+--model "$3"}
    else
        echo "Found $1; skipping (use --force to rebuild)."
    fi
}

neural_retrieval() {
    # neural_retrieval <dense|sparse> <run name> <index file> <query sets> [encoder]
    local query_set collection
    for query_set in $4; do
        for collection in paper paragraph; do
            echo "--- $2 / $collection / $query_set"
            python scripts/run_retrieval.py --collection "$collection" --queries "$query_set" \
                --model "$1" --run-name "$2" --index "$3" ${5:+--checkpoint "$5"}
        done
    done
}

# Conda's activation scripts reference unset variables, so pause set -u briefly
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# ---- Experiments ----
run_experiments() {
    step "Checking inputs"
    local inputs=("data/raw/JSON Files" data/raw/Study.json data/raw/Test.json
                  data/raw/Study_paper_qrel.tsv data/raw/Study_paragraph_qrel.tsv
                  config/bm25.yaml config/preprocessing_neural.yaml)
    if $TRAIN; then
        inputs+=(data/raw/Train.json data/raw/Train_paragraph_qrel.tsv)
    fi
    for path in "${inputs[@]}"; do
        require "$path" "Run scripts/install.sh, or see the README for where the data goes."
    done
    if ! $TRAIN; then
        for path in "$SPARSE_CHECKPOINT" "$DENSE_CHECKPOINT"; do
            require "$path" "Copy the fine-tuned checkpoints into checkpoints/, or rerun with --train."
        done
    fi
    echo "All inputs found."
    if ! python -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
        echo "Warning: no GPU found; training and indexing will be very slow."
    fi

    step "1/6 Building collections and BM25 indexes"
    # build_collections.py renames both files into place at the very end, so
    # paragraphs.jsonl existing means the build finished.
    local rebuilt=false
    if $FORCE || [ ! -f "$PROCESSED/paragraphs.jsonl" ]; then
        python scripts/build_collections.py
        rebuilt=true
    else
        echo "Found $PROCESSED/paragraphs.jsonl; skipping (use --force to rebuild)."
    fi
    # The indexer writes the paragraph index last, so its presence means both exist
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
    # Cleaned paragraph text for the neural models (math removed, no stemming)
    local neural_rebuilt=false
    if $FORCE || [ ! -f "$NEURAL/paragraphs.jsonl" ]; then
        python scripts/build_text_collections.py
        neural_rebuilt=true
    else
        echo "Found $NEURAL/paragraphs.jsonl; skipping (use --force to rebuild)."
    fi

    step "2/6 Fine-tuning"
    # A new checkpoint or collection means its index must be rebuilt
    local rebuild_all=false
    if $FORCE || $neural_rebuilt; then
        rebuild_all=true
    fi
    local rebuild_sparse=$rebuild_all rebuild_dense=$rebuild_all
    if $TRAIN; then
        # Hard negatives come from BM25's ranking for each Train query
        echo "--- bm25 / paragraph / train (for hard negatives)"
        python scripts/run_retrieval.py --collection paragraph --queries train --model bm25
        python scripts/build_training_data.py
        train_model sparse "$SPARSE_RUN"
        train_model dense "$DENSE_RUN"
        rebuild_sparse=true
        rebuild_dense=true
    else
        echo "Using $SPARSE_CHECKPOINT and $DENSE_CHECKPOINT (use --train to fine-tune)."
    fi

    step "3/6 Building neural paragraph indexes"
    build_sparse_index "$SPARSE_INDEX" "$rebuild_sparse" "$SPARSE_CHECKPOINT"
    build_dense_index "$DENSE_INDEX" "$rebuild_dense" "$DENSE_CHECKPOINT"
    if $ZEROSHOT; then
        build_sparse_index "$SPARSE_ZEROSHOT_INDEX" "$rebuild_all"
        build_dense_index "$DENSE_ZEROSHOT_INDEX" "$rebuild_all"
    fi

    step "4/6 Running retrieval"
    # BM25 reads its tuned k1 and b for each collection from config/bm25.yaml.
    # The neural models index paragraphs only and rank papers by their best paragraph (MaxP).
    local collection
    for collection in paper paragraph; do
        echo "--- bm25 / $collection / study"
        python scripts/run_retrieval.py --collection "$collection" --queries study --model bm25
    done
    neural_retrieval sparse sparse "$SPARSE_INDEX" "study test" "$SPARSE_CHECKPOINT"
    neural_retrieval dense dense "$DENSE_INDEX" "study test" "$DENSE_CHECKPOINT"
    if $ZEROSHOT; then
        neural_retrieval sparse sparse_zeroshot "$SPARSE_ZEROSHOT_INDEX" "study"
        neural_retrieval dense dense_zeroshot "$DENSE_ZEROSHOT_INDEX" "study"
    fi

    step "5/6 Evaluating Study runs"
    mkdir -p results
    python -m src.eval.evaluation --queries study --models bm25 dense sparse \
        | tee results/study_eval_main.txt
    python -m src.eval.evaluation --queries study \
        --models bm25 sparse rrf:bm25+sparse rrf:bm25+dense+sparse \
        | tee results/study_eval_rrf3.txt
    if $ZEROSHOT; then
        python -m src.eval.evaluation --queries study --models bm25 sparse_zeroshot sparse \
            | tee results/study_eval_finetuned.txt
        python -m src.eval.evaluation --queries study --models bm25 dense_zeroshot dense \
            | tee results/study_eval_dense_finetune.txt
    fi

    step "6/6 Checking run files and copying them to \"$RESULTS_DIR\""
    mkdir -p "$RESULTS_DIR"
    local query_set model run_file
    for query_set in study test; do
        for collection in paper paragraph; do
            for model in dense sparse; do
                run_file=$(run_path "$collection" "$model" "$query_set")
                require "$run_file" "Retrieval did not produce it."
                # Every line must have 6 tab-separated columns, with at most 100 results per query
                if ! awk -F'\t' 'NF != 6 { bad = 1 }
                                 { count[$1]++ }
                                 END { for (q in count) if (count[q] > 100) bad = 1; exit bad }' \
                        "$run_file"; then
                    echo "Format problem in $run_file" >&2
                    exit 1
                fi
                echo "OK $(basename "$run_file") ($(cut -f1 "$run_file" | sort -u | wc -l | tr -d ' ') queries)"
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
        if python -c "import json, urllib.request; \
health=json.load(urllib.request.urlopen('http://$HOST:$BACKEND_PORT/api/health', timeout=1)); \
required_models = {'bm25', 'tfidf'}; required_collections = {'paper', 'paragraph'}; \
assert health.get('status') == 'ok' and required_models.issubset(set(health.get('models', []))) \
and required_collections.issubset(set(health.get('collections', [])))" 2>/dev/null; then
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
