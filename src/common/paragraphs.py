"""
Reading the cleaned-text paragraph collection
(data/processed_neural/paragraphs.jsonl).

Both neural models read the collection the same way: in batches when encoding the 
whole collection into an index, and by ID when building training batches. 
Use the same with_title, setting for training and indexing, so the model sees 
paragraphs in the same form both times.

Usage:
    from src.common.paragraphs import iter_paragraph_batches, load_paragraph_texts

    for doc_ids, texts in iter_paragraph_batches(path, batch_size=256):
        vectors = encoder.encode(texts)
    
    texts = load_paragraph_texts(path, doc_ids={"2409.10230_35", "2408.02148_9"})
"""
import json

def format_paragraph(record, with_title=False):
    """
    Return the text a model should encode for one paragraph.

    Args:
        record (dict): One line of paragraphs.jsonl.
        with_title (bool): Put the paper title before the paragraph text.
    
    Returns:
        str: The text to encode.
    """
    if with_title and record["title"]:
        return f"{record['title']}. {record['text']}"
    return record["text"]

def iter_paragraph_batches(paragraphs_path, batch_size=256, with_title=False, limit=None):
    """
    Read paragraphs.jsonl in batches, in file order.

    Reads one line at a time, so memory use stays at one batch however large the collection 
    is. Index rows should follow this order, so row i of an index is the i-th paragraph in the file.

    Args: 
        paragraphs_path (str or Path): Path to paragraphs.jsonl.
        batch_size (int): Paragraphs per batch; the last batch may be smaller.
        with_title (bool): Put the paper title before each paragraph's text.
        limit (int or None): Stop after this many paragraphs (for quick tests).

    Yields:
        tuple: (doc_ids, texts), two lists of the same length.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size if {batch_size!r}; expected 1 or more.")

    doc_ids, texts = [], []
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if limit is not None and line_number >= limit:
                break
            record = json.loads(line)
            doc_ids.append(record["doc_id"])
            texts.append(format_paragraph(record, with_title))
            if len(doc_ids) == batch_size:
                yield doc_ids, texts
                doc_ids, texts = [], []
    if doc_ids:
        yield doc_ids, texts

def load_paragraph_texts(paragraphs_path, doc_ids=None, with_title=False):
    """
    Load paragraph texts by ID.

    All 2.27M paragraphs take several GB of memory, so pass doc_ids to keep 
    only the ones needed (e.g., the paragraphs in the training examples).

    Args:
        paragraphs_path (str or Path): Path to paragraphs.jsonl.
        doc_ids (iterable): IDs to keep; None keeps every paragraph.
        with_title (bool): Put the paper title before each paragraph's text.
    
    Returns:
        dict: doc_id -> text to encode.
    
    Raises:
        KeyError: If any requested ID is not in the collection.
    """
    wanted = None if doc_ids is None else set(doc_ids)
    texts = {}
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if wanted is None or record["doc_id"] in wanted:
                texts[record["doc_id"]] = format_paragraph(record, with_title)

    if wanted is not None:
        missing = wanted - texts.keys()
        if missing:
            raise KeyError(f"{len(missing):,} paragraph IDs are not in {paragraphs_path}, "
                           f"e.g., {sorted(missing)[:3]}")
    return texts