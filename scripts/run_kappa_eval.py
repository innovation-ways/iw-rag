"""Judge–human agreement (Cohen's κ) for the faithfulness judge — LIVE.

The one Phase-1 trust number still owed: how well the local offline judge
(Lynx-8B) agrees with human ground truth. This runs the served judge over the
calibration slice (``eval/calibration/faithfulness_slice.yaml``), pairs each
verdict with the slice's ``human_label``, and computes Cohen's κ via
:func:`iw_rag.eval.calibration.judge_human_agreement`.

Workflow for the human in the loop:
  1. Open eval/calibration/faithfulness_slice.yaml and confirm/edit each
     ``human_label`` (the committed labels are *proposed* ground truth).
  2. Bring up a faithfulness judge at
     ``IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT`` (Lynx-8B). Note: the judge
     is an 8B model — free VRAM first (e.g. stop the guard model) since guards-on
     already sits near the 32 GB ceiling (two-clocks: the judge is offline).
  3. Run::

         uv run --group eval-judges python scripts/run_kappa_eval.py

κ = 1.0 perfect · 0.0 chance · <0 worse-than-chance. A `None` result is the
"uncalibrated" sentinel (degenerate label distribution).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from iw_rag.eval.calibration import judge_human_agreement
from iw_rag.eval.judges import FaithfulnessJudge
from iw_rag.generation.llm import VllmClient
from iw_rag.settings import get_settings

_SLICE = Path(__file__).resolve().parent.parent / "eval" / "calibration" / "faithfulness_slice.yaml"


async def main() -> None:
    cfg = get_settings()
    gen = cfg.models.get(cfg.generation.model)
    judge_cfg = cfg.models.get(cfg.eval.judges.faithfulness_model)
    judge = FaithfulnessJudge(
        judge_client=VllmClient(judge_cfg.endpoint),
        generator_model=gen.served_name,
        judge_model=judge_cfg.served_name,
    )

    spec = yaml.safe_load(_SLICE.read_text())
    examples = spec["examples"]
    print(
        f"Judge–human κ over {len(examples)} calibration examples (judge={judge_cfg.served_name})\n"
    )

    human_labels: list[bool] = []
    judge_labels: list[bool] = []
    for ex in examples:
        verdict = await judge.judge(ex["question"], ex["context"], ex["answer"])
        human = bool(ex["human_label"])
        human_labels.append(human)
        judge_labels.append(verdict.passed)
        agree = "✓" if verdict.passed == human else "✗ DISAGREE"
        print(
            f"  [{ex['id']}] human={'FAITHFUL' if human else 'UNFAITHFUL':10} "
            f"judge={'FAITHFUL' if verdict.passed else 'UNFAITHFUL':10} {agree}"
        )

    kappa = judge_human_agreement(judge_labels, human_labels)
    n_agree = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h)
    print(f"\n  raw agreement: {n_agree}/{len(examples)}")
    if kappa is None:
        print("  Cohen's κ: UNCALIBRATED (degenerate label distribution)")
    else:
        print(f"  Cohen's κ: {kappa:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
