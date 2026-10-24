"""Chunking lab: run IW-RAG's real chunker over a sample and report what it did.

The lab uses the production :class:`~iw_rag.ingestion.chunk.HybridChunker` with the
online path's character tokenizer (one token is one character), so a leaf size here
means characters, exactly as for a document uploaded in the console.  Alongside the
chunks it reports the small-to-big expansion rule the retrieval spine applies and
whether each probe question's answer sentence survived intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iw_rag.ingestion.chunk import ChunkConfig, ChunkTree, HybridChunker
from iw_rag.ingestion.light_parse import CharTokenizer
from iw_rag.ingestion.parse import parse_markdown_text
from iw_rag.learn.samples import Probe, Sample

# Mirrors RetrievalSettings.parent_expansion_max_chars: a parent larger than this is
# not handed to the generator; the matched leaf is used instead (spine._expand_parents).
PARENT_EXPANSION_MAX_CHARS = 4000

MIN_LEAF_CHARS = 50
MAX_LEAF_CHARS = 2500

_SENTENCE_END = (".", "!", "?", ":", "|")
_MAX_SPAN = 6

ProbeStatus = Literal["intact", "bare", "split", "missing"]


@dataclass(frozen=True, slots=True)
class LabLeaf:
    """One leaf chunk, as search would see it."""

    index: int
    parent_index: int
    section: str
    chars: int
    text: str
    content_type: str
    ends_mid_sentence: bool


@dataclass(frozen=True, slots=True)
class LabParent:
    """One parent (section) chunk, as the generator would see it."""

    index: int
    section: str
    chars: int
    leaf_count: int
    expanded: bool


@dataclass(frozen=True, slots=True)
class LabProbe:
    """What happened to one question's answer sentence."""

    question: str
    sentence: str
    status: ProbeStatus
    leaf_indexes: tuple[int, ...]
    model_reads_chars: int | None
    verdict: str
    detail: str


@dataclass(frozen=True, slots=True)
class ChunkingLabResult:
    """Everything the chunking lab shows for one sample and one leaf size."""

    sample_id: str
    leaf_chars: int
    parent_budget_chars: int
    leaves: tuple[LabLeaf, ...]
    parents: tuple[LabParent, ...]
    probes: tuple[LabProbe, ...]
    avg_leaf_chars: int
    mid_sentence_cuts: int
    oversized_parents: int


def _ends_mid_sentence(text: str, content_type: str) -> bool:
    if content_type != "text":
        return False
    return not text.rstrip().endswith(_SENTENCE_END)


def _locate(leaves: list[LabLeaf], sentence: str) -> tuple[int, int] | None:
    """Find the leaves holding ``sentence``: ``(first, last)`` indexes, or ``None``.

    Leaves of one block are contiguous slices of its text, so a sentence cut by a
    boundary reappears when the neighbouring leaves are joined.  The first leaf must
    hold the sentence's start, which keeps a sentence that sits wholly in a later
    leaf from being attributed to an earlier one.
    """
    for first in range(len(leaves)):
        joined = ""
        for last in range(first, min(first + _MAX_SPAN, len(leaves))):
            joined += leaves[last].text
            if sentence not in joined:
                continue
            tail = "".join(leaf.text for leaf in leaves[first + 1 : last + 1])
            if last == first or sentence not in tail:
                return first, last
    return None


def _probe(probe: Probe, leaves: list[LabLeaf], parents: list[LabParent]) -> LabProbe:
    found = _locate(leaves, probe.sentence)
    if found is None:
        return LabProbe(
            probe.question,
            probe.sentence,
            "missing",
            (),
            None,
            "Answer not found",
            "No leaf holds it.",
        )
    first, last = found
    if first == last:
        leaf = leaves[first]
        parent = parents[leaf.parent_index]
        if parent.expanded:
            return LabProbe(
                probe.question,
                probe.sentence,
                "intact",
                (first,),
                parent.chars,
                "Answer kept in one leaf",
                f"Search matches leaf {first + 1}. The model then reads its whole section "
                f"({parent.chars:,} characters), which gives it context around the answer.",
            )
        return LabProbe(
            probe.question,
            probe.sentence,
            "bare",
            (first,),
            leaf.chars,
            "Answer kept whole, but with no surrounding context",
            f"Search matches leaf {first + 1}, but its section is {parent.chars:,} characters, "
            f"over the {PARENT_EXPANSION_MAX_CHARS:,} budget, so the model reads only the leaf "
            f"({leaf.chars:,} characters) with no surrounding context.",
        )
    span = tuple(range(first, last + 1))
    names = " and ".join(str(i + 1) for i in span)
    return LabProbe(
        probe.question,
        probe.sentence,
        "split",
        span,
        None,
        f"Answer cut across leaves {names}",
        "No single leaf contains the whole answer, so a search for it matches a fragment. "
        "The parent section may still rescue it if it is expanded, "
        "but the match itself is weakened.",
    )


def run_chunking_lab(sample: Sample, leaf_chars: int) -> ChunkingLabResult:
    """Chunk ``sample`` with the real chunker at ``leaf_chars`` and describe the outcome.

    Args:
        sample: The sample document.
        leaf_chars: Target leaf size in characters, clamped to the lab's range.

    Returns:
        The leaves, parents, per-probe verdicts and summary numbers.
    """
    leaf_chars = max(MIN_LEAF_CHARS, min(MAX_LEAF_CHARS, leaf_chars))
    doc = parse_markdown_text(sample.text, f"learn://{sample.sample_id}")
    tree: ChunkTree = HybridChunker(ChunkConfig(leaf_tokens=leaf_chars), CharTokenizer()).chunk(doc)

    parent_index = {node.local_id: i for i, node in enumerate(tree.parents)}
    leaves = [
        LabLeaf(
            index=i,
            parent_index=parent_index[node.parent_local_id or ""],
            section=node.section_path or "(no heading)",
            chars=len(node.content),
            text=node.content,
            content_type=node.content_type,
            ends_mid_sentence=_ends_mid_sentence(node.content, node.content_type),
        )
        for i, node in enumerate(tree.leaves)
    ]
    counts = [0] * len(tree.parents)
    for leaf in leaves:
        counts[leaf.parent_index] += 1
    parents = [
        LabParent(
            index=i,
            section=node.section_path or "(no heading)",
            chars=len(node.content),
            leaf_count=counts[i],
            expanded=len(node.content) <= PARENT_EXPANSION_MAX_CHARS,
        )
        for i, node in enumerate(tree.parents)
    ]
    return ChunkingLabResult(
        sample_id=sample.sample_id,
        leaf_chars=leaf_chars,
        parent_budget_chars=PARENT_EXPANSION_MAX_CHARS,
        leaves=tuple(leaves),
        parents=tuple(parents),
        probes=tuple(_probe(p, leaves, parents) for p in sample.probes),
        avg_leaf_chars=round(sum(x.chars for x in leaves) / len(leaves)) if leaves else 0,
        mid_sentence_cuts=sum(1 for x in leaves if x.ends_mid_sentence),
        oversized_parents=sum(1 for p in parents if not p.expanded),
    )


__all__ = [
    "MAX_LEAF_CHARS",
    "MIN_LEAF_CHARS",
    "PARENT_EXPANSION_MAX_CHARS",
    "ChunkingLabResult",
    "LabLeaf",
    "LabParent",
    "LabProbe",
    "run_chunking_lab",
]
