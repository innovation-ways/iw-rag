"""Generate a long (~22K-token) single document with a buried needle (exp #8).

A NoLiMa/RULER-style probe doc: plausible 'employee handbook' filler with ONE
unique needle fact planted at ~60% depth. The needle uses entities/wording that do
NOT lexically overlap the question, so top-k retrieval is prone to miss it while a
native-32K whole-doc pass sees it. Deterministic (no RNG, no LLM).

Usage:  uv run python scripts/gen_longdoc.py --out data/longdoc_corpus
Then ingest (CPU): CUDA_VISIBLE_DEVICES="" make ingest CORPUS=data/longdoc_corpus/manifest.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# The needle: a unique fact with wording deliberately distinct from the question
# "What is the recovery passphrase for the archive vault?" → answer below.
NEEDLE = (
    "Operational footnote 7731: the long-term cold-storage enclosure in sub-level B "
    "is unsealed only with the custodial recovery passphrase **mauve-otter-49-banister**, "
    "memorised by the duty archivist and never written in any ticket."
)
NEEDLE_ANSWER = "mauve-otter-49-banister"
NEEDLE_QUESTION = "What is the recovery passphrase for the archive vault?"

# Filler paragraph templates — generic handbook prose, no overlap with the needle.
_TOPICS = [
    "meeting-room booking etiquette and the shared calendar conventions",
    "the quarterly all-hands logistics and the cafeteria rotation schedule",
    "laptop refresh cycles and the asset-tagging workflow for new hires",
    "the visitor sign-in procedure and badge colours for contractors",
    "expense pre-approval thresholds for conference travel and per-diem rounding",
    "the bicycle-storage cage rules and the shower-facility booking window",
    "internal newsletter cadence and the style guide for announcements",
    "desk-ergonomics assessments and the standing-desk request backlog",
    "the printer-credit allowance and the recycling-bin colour coding",
    "parking-permit lottery rules and the electric-vehicle charging queue",
]


def _para(i: int) -> str:
    topic = _TOPICS[i % len(_TOPICS)]
    return (
        f"Section {i + 1}. This portion of the handbook concerns {topic}. "
        "Employees are reminded that the relevant forms live in the intranet under the "
        "Operations space, and that any exceptions require sign-off from the floor "
        "coordinator before the end of the affected week. Historically these processes "
        "were paper-based, but they have since moved to the self-service portal, which "
        "tracks each request through draft, review, and resolution states. The facilities "
        "team reviews the aggregate metrics monthly and publishes a short summary so that "
        "teams can plan around peak demand. When in doubt, prefer the asynchronous channel "
        "and attach the reference number so the audit trail stays complete and searchable."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/longdoc_corpus")
    ap.add_argument("--paragraphs", type=int, default=140)  # ~20K tokens (fits 32K + query/output)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ops").mkdir(parents=True, exist_ok=True)

    paras = [_para(i) for i in range(args.paragraphs)]
    needle_at = int(len(paras) * 0.6)
    paras.insert(needle_at, NEEDLE)
    body = "# Operations Handbook (Consolidated)\n\n" + "\n\n".join(paras) + "\n"

    doc_path = out / "ops" / "handbook.md"
    doc_path.write_text(body)

    manifest = {
        "documents": [
            {
                "uri": "corpus://longdoc/ops/handbook.md",
                "path": "ops/handbook.md",
                "mime": "text/markdown",
                "acl_groups": ["public"],
            }
        ]
    }
    (out / "manifest.yaml").write_text(
        "# Long-doc needle corpus (exp #8). Public/synthetic.\n"
        + yaml.safe_dump(manifest, sort_keys=False)
    )

    words = len(body.split())
    toks = int(words / 0.75)
    print(f"Wrote {doc_path} — {len(paras)} paragraphs, ~{words} words (~{toks} tokens)")
    print(f"Needle at paragraph {needle_at} (~60% depth). Answer: {NEEDLE_ANSWER!r}")
    print(f"Question: {NEEDLE_QUESTION!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
