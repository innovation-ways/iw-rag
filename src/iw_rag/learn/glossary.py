"""Glossary for the Learn section, in the terms IW-RAG itself uses."""

from __future__ import annotations

GLOSSARY: tuple[tuple[str, str], ...] = (
    (
        "ACL",
        "Access control list: the groups allowed to read a document. In IW-RAG it is applied "
        "as a pre-filter inside the search, never after it.",
    ),
    (
        "principal_groups",
        "The groups a signed-in user belongs to. They come from authentication, "
        "never from the question or the model.",
    ),
    (
        "Leaf and parent",
        "A leaf is the small chunk that is embedded and searched; its parent is "
        "the section that contains it, handed to the model for context.",
    ),
    (
        "Small-to-big",
        "Match on small leaves for precision, answer from their larger parents for context.",
    ),
    (
        "Late chunking",
        "Embedding the whole document in one pass and taking each chunk's vector "
        "from that pass, so chunks keep document-level context.",
    ),
    ("Embedding", "A list of numbers (1024 for BGE-M3) representing the meaning of a text."),
    ("BM25", "Classic keyword scoring. It is the exact-word arm of IW-RAG's hybrid search."),
    (
        "RRF",
        "Reciprocal Rank Fusion: merges ranked lists using only each item's rank (k = 60 here).",
    ),
    (
        "Cross-encoder reranker",
        "A model that reads a question and a chunk together and scores "
        "how well the chunk answers it.",
    ),
    (
        "top-k",
        "How many results are kept at a stage. IW-RAG keeps 40 per search arm, 80 after "
        "fusion, and 8 after reranking.",
    ),
    (
        "Spine",
        "IW-RAG's retrieval pipeline: access filter, hybrid search, fusion, rerank, expansion.",
    ),
    (
        "Datamarking",
        "Weaving an invisible marker through retrieved text so the model can tell "
        "data from instructions.",
    ),
    (
        "Indirect injection",
        "Hidden instructions planted in a document that try to hijack the model "
        "when the document is retrieved.",
    ),
    ("Grounded", "Every claim in an answer is supported by the retrieved passages."),
    (
        "Fail closed",
        "When unsure, refuse instead of guessing. The default for every guard in IW-RAG.",
    ),
    (
        "Verdict trail",
        "The ordered list of decisions (route, grades, guards) recorded for each question.",
    ),
    (
        "Adaptive routing",
        "Choosing per question whether to skip retrieval, search, or use an optional mode.",
    ),
    (
        "Golden set",
        "A fixed set of questions with known correct sources, used to measure the system.",
    ),
    ("Recall@k", "The share of questions where a correct passage appears in the top k results."),
    ("MRR", "Mean Reciprocal Rank: the average of 1 / (position of the first correct result)."),
    (
        "nDCG",
        "A score for how well the whole ranked list is ordered, rewarding correct results "
        "near the top.",
    ),
    ("Faithfulness", "How well an answer sticks to its sources, with no invented claims."),
    ("Cohen's kappa", "Agreement between the judge model and human labels, corrected for chance."),
    (
        "Serving and benchmark mode",
        "The single GPU does one job at a time: answering questions, or "
        "running model-swap benchmarks.",
    ),
    ("VRAM", "The GPU's own memory. The generator, embedder and reranker must fit in it together."),
)

__all__ = ["GLOSSARY"]
