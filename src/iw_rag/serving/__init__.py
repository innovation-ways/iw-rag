"""Benchmark Studio model-swap serving package (F-00094).

This package is the **single swap-state authority** for the Benchmark Studio
generator-swap infrastructure.  It contains:

* :mod:`iw_rag.serving.catalog` — the YAML catalog loader + Pydantic models
  for the user-facing generator catalog.
* :mod:`iw_rag.serving.llama_swap_client` — the typed async HTTP client that
  talks to the local ``llama-swap`` proxy.
* :mod:`iw_rag.serving.swap_controller` — the activation state machine that
  loads the catalog, triggers a swap via llama-swap, and polls the model's
  readiness endpoint until it reports HTTP 200.
* :mod:`iw_rag.serving.swap_controller_client` — the typed async HTTP client
  the main API uses to proxy to this sidecar (``GET /catalog``,
  ``POST /catalog/activate``, ``GET /catalog/status``).
* :mod:`iw_rag.serving.app` — the thin FastAPI sidecar app that owns the
  single ``SwapController`` instance.

**Architectural boundary.**  ``serving/`` may import ``settings``,
``domain.errors``, and stdlib + httpx; it MUST NOT import ``api/``,
``agent/``, or ``storage/`` (dependency cone stays inward).  The public API
(``src/iw_rag/api/``) is the only consumer that imports ``serving/``; the
sidecar itself is internal-only and never imported by the main API process
(it is reached over HTTP at ``http://swap-controller:8090``).

**Single swap-state authority.**  The :class:`~.swap_controller.SwapController`
is instantiated exactly once per process — inside this sidecar.  The main
API process holds NO in-process ``SwapController``; it proxies to the
sidecar via :class:`~.swap_controller_client.SwapControllerClient`.  This
boundary avoids split-brain swap state (two processes each holding their
own ``swapping`` flag and neither seeing the other's in-flight swap, which
would break the AC4 / Boundary-row-4 503 guard).
"""

from __future__ import annotations
