"""Experiment #8 deepening — NoLiMa/RULER-style needle-depth sweep (LIVE).

The #8 probe (`run_longcontext_eval.py`) places one needle at 60% depth. This sweeps
the needle across the document and measures native-32K recovery at each depth — the
"lost in the middle" curve for Qwen3-14B at ~21K tokens. The document is built
in-memory per depth (the long-context arm stuffs the whole doc, so no re-ingest is
needed; the question/needle are identical, only the position moves).

Requires the generator at native 32K (`--max-model-len 32768`); see
`run_longcontext_eval.py` for the vLLM launch. Run:

    uv run python scripts/run_longcontext_sweep.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `gen_longdoc` imports

# Reuse the exact needle + filler from the #8 corpus generator for parity.
from gen_longdoc import (  # noqa: E402
    NEEDLE,
    NEEDLE_ANSWER,
    NEEDLE_QUESTION,
    _para,
)

from iw_rag.generation.llm import GenerationParams, VllmClient  # noqa: E402
from iw_rag.settings import get_settings  # noqa: E402

_DEPTHS = [0.05, 0.25, 0.50, 0.75, 0.95]
_N_PARAS = 140
# Context-length axis: paragraph counts → ~9K / 18K / 26K / 31K tokens (the last is
# right at the native 32K ceiling — the effective-length cliff probe).
_LENGTHS = [60, 120, 175, 205]
_OUT = "eval/reports/longcontext_depth_sweep.json"
_OUT_LEN = "eval/reports/longcontext_length_sweep.json"
_SYSTEM = (
    "You answer strictly from the supplied document. If the document contains the "
    "answer, quote the exact value."
)


def _doc(n_paras: int, depth: float) -> str:
    paras = [_para(i) for i in range(n_paras)]
    paras.insert(int(len(paras) * depth), NEEDLE)
    return "# Operations Handbook (Consolidated)\n\n" + "\n\n".join(paras) + "\n"


def _doc_with_needle_at(depth: float) -> str:
    return _doc(_N_PARAS, depth)


def _toks(text: str) -> int:
    return int(len(text.split()) / 0.75)


def _write(path: str, report: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))


async def _recover(llm: VllmClient, params: GenerationParams, doc: str) -> tuple[bool, float]:
    t0 = time.monotonic()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{doc}\n\nQuestion: {NEEDLE_QUESTION} /no_think"},
    ]
    try:
        resp = await llm.complete(messages, params)
    except Exception as exc:  # noqa: BLE001 — an over-window doc 400s; that's a data point
        print(f"     (generation failed: {str(exc)[:80]})")
        return False, (time.monotonic() - t0) * 1000.0
    return NEEDLE_ANSWER.lower() in resp.text.lower(), (time.monotonic() - t0) * 1000.0


async def _depth_sweep(llm: VllmClient, params: GenerationParams) -> None:
    print(f"#8 needle-DEPTH sweep · native-32K · needle={NEEDLE_ANSWER!r}\n")
    rows = []
    for depth in _DEPTHS:
        doc = _doc(_N_PARAS, depth)
        found, ms = await _recover(llm, params, doc)
        rows.append(
            {
                "depth_pct": int(depth * 100),
                "context_tokens_est": _toks(doc),
                "needle_recovered": found,
                "latency_ms": round(ms, 1),
            }
        )
        print(
            f"  depth {int(depth * 100):>3}%  recovered={'YES' if found else 'NO '}  "
            f"ctx≈{_toks(doc)}tok  {ms:.0f}ms"
        )
    n_ok = sum(1 for r in rows if r["needle_recovered"])
    _write(
        _OUT,
        {
            "experiment": "exp8_longcontext_depth_sweep",
            "needle": NEEDLE_ANSWER,
            "recovered_count": f"{n_ok}/{len(rows)}",
            "per_depth": rows,
        },
    )
    print(f"\nRecovered at {n_ok}/{len(rows)} depths. Wrote {_OUT}")


async def _length_sweep(llm: VllmClient, params: GenerationParams) -> None:
    print(f"#8 context-LENGTH sweep · needle at 50% · needle={NEEDLE_ANSWER!r}\n")
    rows = []
    for n in _LENGTHS:
        doc = _doc(n, 0.5)
        found, ms = await _recover(llm, params, doc)
        rows.append(
            {
                "n_paragraphs": n,
                "context_tokens_est": _toks(doc),
                "needle_recovered": found,
                "latency_ms": round(ms, 1),
            }
        )
        print(
            f"  {n:>3} paras  ctx≈{_toks(doc):>5}tok  recovered={'YES' if found else 'NO '}  "
            f"{ms:.0f}ms"
        )
    n_ok = sum(1 for r in rows if r["needle_recovered"])
    _write(
        _OUT_LEN,
        {
            "experiment": "exp8_longcontext_length_sweep",
            "needle": NEEDLE_ANSWER,
            "needle_depth_pct": 50,
            "recovered_count": f"{n_ok}/{len(rows)}",
            "per_length": rows,
        },
    )
    print(f"\nRecovered at {n_ok}/{len(rows)} lengths. Wrote {_OUT_LEN}")


async def main(axis: str) -> None:
    cfg = get_settings()
    gen = cfg.models.get(cfg.generation.model)
    llm = VllmClient(gen.endpoint)
    params = GenerationParams(model=gen.served_name, temperature=0.0, max_tokens=128)
    if axis in ("depth", "both"):
        await _depth_sweep(llm, params)
    if axis in ("length", "both"):
        await _length_sweep(llm, params)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["depth", "length", "both"], default="depth")
    asyncio.run(main(ap.parse_args().axis))
