"""
Tokenization module for the IR-Rangers retrieval system.

Turns text into tokens, both when building the index offline and when
processing queries at search time. Documents and queries must be tokenized
with the same configuration, or matching terms will not line up.

Pipeline, in order:
    1. Clean the raw text (inline math, URLs, citation markers)
    2. Lowercase
    3. Split into tokens of letters and digits
    4. Filter tokens (minimum length, numbers-only, stopwords)
    5. Stem

Settings are read from config/preprocessing.yaml.

Usage:
    from src.preprocessing.tokenizer import tokenize_text
    tokens = tokenize_text("Your text here")

    or, from the repository root:

    python -m src.preprocessing.tokenizer "<your text here>"
"""
import re
import sys
from pathlib import Path
from functools import lru_cache

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocessing.yaml"

# Cleaning patterns, applied to raw text before tokenizing.
MATH_SPAN = re.compile(r"\$[^$]*\$")                            # $f_{\mathcal{T}}$
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")                      # \ldots outside of $...$
URL = re.compile(r"https?://\S+|www\.\S+")
CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")      # [1], [5, 6, 7], [12-15]

# Letters and digits only, so underscores split tokens (x_i -> x, i).
TOKEN_PATTERN = re.compile(r"[^\W_]+")

STOPWORD_SOURCES = {"nltk", "custom", "none"}
STEMMERS = {"none", "porter", "snowball"}

CLEANING_KEYS = {"remove_math", "remove_urls", "remove_citations"}
TOKENIZER_KEYS = {"lowercase", "min_token_length", "drop_numeric", "stopwords", "stemming"}


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Read a preprocessing config file.

    Args:
        config_path (str or Path): Path to the YAML config file.

    Returns:
        dict: The whole config, with "cleaning" and "tokenizer" sections.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _check_keys(section, allowed, name):
    """Reject unknown config keys, so a typo like "remove_url" fails instead of being ignored."""
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{name}' config: {sorted(unknown)}. "
            f"Expected some of: {sorted(allowed)}."
        )

def clean_text(text, remove_math=False, remove_urls=False, remove_citations=False):
    """
    Remove noise from raw text before tokenizing.

    Each match is replaced with a space so the words on either side do not merge.

    Args:
        text (str): Raw text from a document or query.
        remove_math (bool): Remove inline $...$ math and leftover LaTeX commands.
        remove_urls (bool): Remove http(s) and www URLs.
        remove_citations (bool): Remove numeric citation markers like [1, 2].

    Returns:
        str: The cleaned text.
    """
    if remove_math:
        # With an odd number of "$", at least one is a plain dollar sign
        # (e.g. "costs $5"), and pairing them up could delete real text.
        if text.count("$") % 2 == 0:
            text = MATH_SPAN.sub(" ", text)
        text = LATEX_COMMAND.sub(" ", text)
    if remove_urls:
        text = URL.sub(" ", text)
    if remove_citations:
        text = CITATION.sub(" ", text)
    return text


def _normalize_word_list(words, name):
    """
    Lowercase and strip a word list from the config.

    An empty YAML list (e.g. "extra:" with nothing after it) loads as None,
    so None is treated as an empty list. Non-string entries are rejected,
    since PyYAML reads unquoted yes/no/on/off as booleans.
    """
    if words is None:
        return set()

    normalized = set()
    for word in words:
        if not isinstance(word, str):
            raise ValueError(
                f"stopwords.{name} contains {word!r}, which is not a string. "
                f'Put it in quotes in the YAML file (e.g. "no" instead of no).'
            )
        normalized.add(word.strip().lower())
    return normalized


def _load_nltk_stopwords():
    """Load NLTK's English stopword list, failing clearly if it is not installed."""
    from nltk.corpus import stopwords

    try:
        return set(stopwords.words("english"))
    except LookupError:
        raise LookupError(
            "NLTK stopwords are not installed. Run `bash install.sh`, or "
            "`python -m nltk.downloader stopwords`."
        ) from None


def _load_custom_stopwords(custom_file):
    """Load a stopword file with one word per line, skipping blank lines."""
    path = REPO_ROOT / custom_file
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

def _build_stemmer(name):
    """
    Return a cached stem function for the configured stemmer, or None for no stemming.

    Stems are cached because the collection has millions of tokens but far
    fewer unique words, so each unique word is only stemmed once.
    """
    if name == "none":
        return None

    from nltk.stem import PorterStemmer, SnowballStemmer

    if name == "porter":
        stemmer = PorterStemmer()
    else:
        stemmer = SnowballStemmer("english")
    return lru_cache(maxsize=None)(stemmer.stem)


def build_stopwords(stopword_config):
    """
    Build the final stopword set from the config.

    Starts from the chosen source list, adds "extra", then removes "keep".

    Args:
        stopword_config (dict): The "stopwords" section of the tokenizer config.

    Returns:
        set: Lowercase stopwords.
    """
    source = stopword_config.get("source")
    if source not in STOPWORD_SOURCES:
        raise ValueError(
            f"stopwords.source is {source!r}; expected one of {sorted(STOPWORD_SOURCES)}."
        )

    if source == "none":
        return set()

    if source == "nltk":
        words = _load_nltk_stopwords()
    else:
        custom_file = stopword_config.get("custom_file")
        if not custom_file:
            raise ValueError("stopwords.source is 'custom' but no custom_file is set.")
        words = _load_custom_stopwords(custom_file)

    words |= _normalize_word_list(stopword_config.get("extra"), "extra")
    words -= _normalize_word_list(stopword_config.get("keep"), "keep")
    return words


class Tokenizer:
    """
    Text-to-tokens pipeline built from one config file.

    Create separate Tokenizer objects to compare configurations in the same script.
    """

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        config = load_config(config_path)
        cleaning_config = config.get("cleaning") or {}
        tokenizer_config = config.get("tokenizer") or {}
        _check_keys(cleaning_config, CLEANING_KEYS, "cleaning")
        _check_keys(tokenizer_config, TOKENIZER_KEYS, "tokenizer")

        self.cleaning = {key: bool(cleaning_config.get(key, False)) for key in CLEANING_KEYS}

        self.lowercase = tokenizer_config.get("lowercase", True)

        self.min_token_length = tokenizer_config.get("min_token_length", 1)
        if not isinstance(self.min_token_length, int) or self.min_token_length < 1:
            raise ValueError(
                f"min_token_length is {self.min_token_length!r}; expected a whole number of 1 or more."
            )

        self.drop_numeric = tokenizer_config.get("drop_numeric", False)

        self.stemming = tokenizer_config.get("stemming", "none")
        if self.stemming not in STEMMERS:
            raise ValueError(
                f"stemming is {self.stemming!r}; expected one of {sorted(STEMMERS)}."
            )
        self.stem = _build_stemmer(self.stemming)

        self.stopwords = build_stopwords(tokenizer_config.get("stopwords") or {})
    
    def tokenize(self, text):
        """
        Clean text, split it into tokens, filter them, and stem them.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: Tokens in their original order.
        """
        text = clean_text(text, **self.cleaning)

        if self.lowercase:
            text = text.lower()
        # Note: stopwords are lowercase, so with lowercase off, capitalized
        # stopwords such as "The" are not removed.

        tokens = [
            token for token in TOKEN_PATTERN.findall(text)
            if len(token) >= self.min_token_length
            and not (self.drop_numeric and token.isdigit())
            and token not in self.stopwords
        ]

        # Stem after stopword removal, since the stopword lists contain unstemmed words.
        if self.stem is not None:
            tokens = [self.stem(token) for token in tokens]

        return tokens


_default_tokenizer = None


def tokenize_text(text):
    """
    Tokenize text with the default config (config/preprocessing.yaml).

    The config is loaded on first use, so importing this module has no side effects.
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()
    return _default_tokenizer.tokenize(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.preprocessing.tokenizer "<your text here>"')
        sys.exit(1)

    print(tokenize_text(" ".join(sys.argv[1:])))