"""eval/redteam — versioned adversarial attack corpus + loader (Task 3.5).

The adversarial corpus is a small, versioned, **synthetic-only** set of:

- Indirect-injection probes (PoisonedRAG / BIPIA / garak latentinjection
  patterns) — payload in retrieved chunks, not in the user turn.
- ACL two-tenant probes — cross-tenant retrieval coercion attempts.
- PII exfiltration probes — synthetic emails/SSNs/phones including
  adversarial paraphrase variants (split turns, base64, "spell it out").
- Unanswerable queries — questions that must force fail-closed refusal.
- Benign queries — used for FPR / FAR computation (must NOT be blocked).

All PII/secrets are **synthetic** (no real data in the repository).
Poisoned docs live in the quarantined ``adversarial`` namespace; they are
**never** ingested into the production index.

Manifest: ``corpus_manifest.yaml`` (versioned alongside the promptfoo config).

References:
- task-3.5-guardrail-eval (spec, §1 adversarial corpus assembly)
- security-red-team-playbook (R-00115) §3 (corpus design rules)
- lld-guardrails §8.4 (attack class → guard layer map)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MANIFEST_PATH = Path(__file__).parent / "corpus_manifest.yaml"


def load_manifest() -> dict[str, Any]:
    """Load and return the adversarial corpus manifest.

    Returns:
        The parsed YAML manifest as a Python dict.

    Raises:
        FileNotFoundError: If ``corpus_manifest.yaml`` is missing.
    """
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Adversarial corpus manifest not found at {_MANIFEST_PATH}. "
            "This file is required for the guardrail red-team evaluation (Task 3.5)."
        )
    with _MANIFEST_PATH.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def get_adversarial_payloads(attack_class: str | None = None) -> list[dict[str, Any]]:
    """Return adversarial payloads from the corpus manifest.

    Args:
        attack_class: Optional filter — returns only payloads from the named
            attack class (e.g. ``"indirect_injection"``, ``"acl_bypass"``,
            ``"pii_exfiltration"``, ``"unanswerable"``).  ``None`` returns
            all payloads across all classes.

    Returns:
        List of payload dicts.  Each dict contains at minimum ``id``,
        ``description``, and ``expected_verdict`` keys.
    """
    manifest = load_manifest()
    payloads: list[dict[str, Any]] = []
    for cls in manifest.get("attack_classes", []):
        if attack_class is not None and cls.get("class") != attack_class:
            continue
        # Payloads may be under "payloads", "queries", or "synthetic_pii_entities"
        for key in ("payloads", "queries", "synthetic_pii_entities"):
            items = cls.get(key, [])
            for item in items:
                payloads.append({"attack_class": cls["class"], **item})
    return payloads


def get_benign_queries() -> list[dict[str, Any]]:
    """Return the benign query set for FPR computation.

    Returns:
        List of benign query dicts (``id``, ``question``,
        ``expected_disposition``).
    """
    manifest = load_manifest()
    return list(manifest.get("benign_queries", []))
