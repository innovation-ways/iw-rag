"""Minimal CPU injection-classifier server for the iw-rag guard_injection role.

Live-validation stand-in for the gated meta-llama Prompt Guard 2: serves the
ungated `protectai/deberta-v3-base-prompt-injection-v2` over the TEI-style
`/classify` contract the input guard expects:

    POST /classify  {"inputs": "<text>"}  ->  [{"label": "...", "score": float}, ...]

The input guard reads the score of the item whose label == "JAILBREAK", so we
map the model's INJECTION class to that label. CPU-only (no VRAM contention).
"""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# INJGUARD_MODEL selects the classifier. The default stays the ungated deberta
# stand-in; set it to the canonical gated `meta-llama/Llama-Prompt-Guard-2-86M`
# (needs HF_TOKEN + accepted Meta licence). Prompt Guard 2 ships id2label
# {LABEL_0: benign, LABEL_1: malicious}, which the index-1 fallback below handles.
_MODEL = os.environ.get("INJGUARD_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")
_tok = AutoTokenizer.from_pretrained(_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL)
_model.eval()
# id2label is typically {0: "SAFE", 1: "INJECTION"}; find the injection index.
_id2label = {int(k): v.upper() for k, v in _model.config.id2label.items()}
_inj_idx = next((i for i, lbl in _id2label.items() if "INJECT" in lbl or "JAIL" in lbl), 1)

app = FastAPI()


class _Req(BaseModel):
    inputs: str


@app.post("/classify")
def classify(req: _Req) -> list[dict[str, object]]:
    """Return TEI-style [{label, score}] with the injection prob under 'JAILBREAK'."""
    enc = _tok(req.inputs, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(_model(**enc).logits, dim=-1)[0]
    p_inj = float(probs[_inj_idx])
    return [
        {"label": "JAILBREAK", "score": p_inj},
        {"label": "BENIGN", "score": float(1.0 - p_inj)},
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
