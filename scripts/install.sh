#!/usr/bin/env bash
# Install everything needed to run the project:
#   - conda (Miniforge), if no conda installation is found
#   - the "ir-rangers" conda environment with the packages in requirements.txt
#   - NLTK stopwords
#   - Node.js (if missing or too old) and the frontend's npm packages
#   - a link to the dataset on the CS servers, if data/raw is missing
#
# Safe to run more than once: each step is skipped when it is already done.
#
# Usage, from anywhere:
#   bash scripts/install.sh
set -euo pipefail

ENV_NAME="ir-rangers"
PYTHON_VERSION="3.8"
MIN_NODE_MAJOR=22                       # Vite 8 needs a recent Node.js
CONDA_INSTALL_DIR="$HOME/miniforge3"    # used only if no conda is found
FRONTEND_DIR="interface/frontend"
SHARED_DATA="/mnt/netstore1_datasets/IR26"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- Helper functions ----
have() {
    command -v "$1" >/dev/null 2>&1
}

find_conda() {
    if have conda; then
        CONDA_BASE="$(conda info --base)"
        return 0
    fi
    for dir in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [ -f "$dir/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$dir"
            return 0
        fi
    done
    return 1
}

install_conda() {
    if ! have curl; then
        echo "Error: curl is needed to download conda. Install curl or conda manually." >&2
        exit 1
    fi
    local installer download_dir
    installer="Miniforge3-$(uname)-$(uname -m).sh"
    download_dir="$(mktemp -d)"
    echo "conda not found. Installing Miniforge into $CONDA_INSTALL_DIR ..."
    curl -fsSL -o "$download_dir/$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
    bash "$download_dir/$installer" -b -p "$CONDA_INSTALL_DIR" # -b: no prompts
    rm -rf "$download_dir"
    CONDA_BASE="$CONDA_INSTALL_DIR"
    echo "Installed. To use conda in new terminals, run: $CONDA_INSTALL_DIR/bin/conda init"
}

node_is_recent() {
    have node && have npm || return 1
    local major
    major="$(node --version | sed 's/^v//' | cut -d. -f1)"
    [ "$major" -ge "$MIN_NODE_MAJOR" ]
}

# Conda's activation scripts reference unset variables, so pause "set -u" around them.
conda_activate() {
    set +u
    conda activate "$1"
    set -u
}

# ---- Conda ----
echo "=== Checking conda ==="
if find_conda; then
    echo "Found conda at $CONDA_BASE"
else
    install_conda
fi
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

# ---- Python Environment ----
echo "=== Setting up the '$ENV_NAME' environment ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    # conda-forge only, so creation doesn't depend on Anaconda's default channel.
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION"
fi
conda_activate "$ENV_NAME"

echo "=== Installing Python packages ==="
# pip skips anything already installed at a matching version.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== Checking NLTK stopwords ==="
if python -c "import nltk; nltk.data.find('corpora/stopwords')" >/dev/null 2>&1; then
    echo "NLTK stopwords already downloaded."
else
    python -c "import nltk; nltk.download('stopwords', quiet=True)"
fi

# ---- Search Interface ----
echo "=== Checking Node.js ==="
NODE_INSTALLED=false
if node_is_recent; then
    echo "Node.js $(node --version) already installed."
else
    # Installed into the conda environment, so the system Node.js is untouched
    echo "Node.js $MIN_NODE_MAJOR+ not found. Installing it into '$ENV_NAME'..."
    conda install -y -n "$ENV_NAME" -c conda-forge --override-channels "nodejs=22"
    hash -r # make the shell look up "node" again
    NODE_INSTALLED=true
fi

echo "=== Installing frontend packages ==="
# node_modules must be installed by the same Node.js that runs it: an older
# Node's npm can skip platform-specific packages (e.g. Vite's native bundler).
# The Node version used for the last install is recorded here.
NODE_STAMP="$FRONTEND_DIR/node_modules/.installed-with-node"
if ! $NODE_INSTALLED && [ -d "$FRONTEND_DIR/node_modules" ] \
        && [ "$(cat "$NODE_STAMP" 2>/dev/null)" = "$(node --version)" ]; then
    echo "Frontend packages already installed with Node.js $(node --version)."
else
    # npm ci removes any existing node_modules, then installs the exact
    # versions in package-lock.json.
    echo "Installing frontend packages with Node.js $(node --version)..."
    (cd "$FRONTEND_DIR" && npm ci)
    node --version > "$NODE_STAMP"
fi

# ---- Data ----
echo "=== Checking data ==="
if [ -e data/raw ]; then
    echo "Found data/raw."
elif [ -d "$SHARED_DATA" ]; then
    mkdir -p data
    ln -s "$SHARED_DATA" data/raw
    echo "Linked data/raw -> $SHARED_DATA"
else
    echo "WARNING: data/raw was not found. Put the dataset there (JSON Files/, query files," >&2
    echo "and QREL files) before running scripts/run.sh." >&2
fi

echo "=== INSTALLATION COMPLETE ==="
echo "Run the experiments with: bash scripts/run.sh"