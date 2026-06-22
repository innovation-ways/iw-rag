"""Unit tests for modes/visual/encoder.py and modes/visual/store.py (WI-031 / Task 4b).

Covers the branches not exercised by existing visual tests:

encoder.py
- ``encode_pages`` failure → ``VisualEncodeUnavailable`` raised with ``match=``
  (any non-``VisualEncodeUnavailable`` exception from the client is wrapped).
- ``encode_pages`` with a client that returns an empty list (empty-result branch).
- ``VisualEncodeUnavailable`` raised by the client is re-raised verbatim
  (not double-wrapped).

store.py
- ``QdrantMultiVectorStore.__init__`` raises ``VisualStoreUnavailable`` when
  ``qdrant_client`` is not installed (ImportError path), with ``match=``.
- ``QdrantMultiVectorStore.__init__`` raises ``VisualStoreUnavailable`` even when
  ``qdrant_client`` *is* importable (the stub seam fires unconditionally), with
  ``match=``.
- ``make_store("postgres_candidates", ...)`` returns a ``PostgresCandidateStore``.
- ``make_store("qdrant", ...)`` propagates ``VisualStoreUnavailable`` from
  ``QdrantMultiVectorStore.__init__``, with ``match=``.
- ``make_store`` with an unknown backend raises ``ValueError``, with ``match=``.

All tests are offline (no live model/network calls).  The Qdrant client is either
absent (real environment) or its import is patched to control the ImportError path.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iw_rag.modes.visual.encoder import ColVisionEncoder, ColVisionEncoderConfig
from iw_rag.modes.visual.errors import VisualEncodeUnavailable, VisualStoreUnavailable
from iw_rag.modes.visual.store import PostgresCandidateStore, make_store

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_encoder_cfg() -> ColVisionEncoderConfig:
    """Build a minimal ColVisionEncoderConfig for tests.

    Returns:
        A ColVisionEncoderConfig with placeholder values (no real endpoint used).
    """
    return ColVisionEncoderConfig(
        model_role="visual_retrieval",
        served_name="vidore/colqwen2.5-v0.2",
        endpoint="http://localhost:9999",
        dims=128,
    )


def _make_encoder(client: MagicMock) -> ColVisionEncoder:
    """Wire a ColVisionEncoder with a fake client.

    Args:
        client: The fake client to inject (its ``encode_pages`` is exercised).

    Returns:
        A ColVisionEncoder pointing at a non-existent endpoint (no network used).
    """
    return ColVisionEncoder(client, _make_encoder_cfg())


# ---------------------------------------------------------------------------
# A. ColVisionEncoder.encode_pages — failure branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_pages_failure_raises_visual_encode_unavailable() -> None:
    """Non-VisualEncodeUnavailable client error is wrapped in VisualEncodeUnavailable."""
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(side_effect=RuntimeError("CUDA OOM"))
    encoder = _make_encoder(fake_client)

    with pytest.raises(VisualEncodeUnavailable, match="encode_pages failed"):
        await encoder.encode_pages([b"fake_image_bytes"])


@pytest.mark.asyncio
async def test_encode_pages_failure_message_contains_original_error() -> None:
    """The VisualEncodeUnavailable message includes the original exception text."""
    original_msg = "GPU memory exhausted"
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(side_effect=OSError(original_msg))
    encoder = _make_encoder(fake_client)

    with pytest.raises(VisualEncodeUnavailable, match=original_msg):
        await encoder.encode_pages([b"img"])


@pytest.mark.asyncio
async def test_encode_pages_value_error_is_wrapped() -> None:
    """A ValueError from the client (unexpected response) is wrapped in VisualEncodeUnavailable."""
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(side_effect=ValueError("unexpected response shape"))
    encoder = _make_encoder(fake_client)

    with pytest.raises(VisualEncodeUnavailable, match="encode_pages failed"):
        await encoder.encode_pages([b"img1", b"img2"])


@pytest.mark.asyncio
async def test_encode_pages_visual_encode_unavailable_is_not_double_wrapped() -> None:
    """VisualEncodeUnavailable raised by the client is re-raised verbatim (no wrapping)."""
    original = VisualEncodeUnavailable("encoder already unavailable")
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(side_effect=original)
    encoder = _make_encoder(fake_client)

    with pytest.raises(VisualEncodeUnavailable, match="encoder already unavailable"):
        await encoder.encode_pages([b"img"])


# ---------------------------------------------------------------------------
# B. ColVisionEncoder.encode_pages — empty-result branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_pages_empty_list_returns_empty_list() -> None:
    """encode_pages with an empty image list returns an empty list (no error)."""
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(return_value=[])
    encoder = _make_encoder(fake_client)

    result = await encoder.encode_pages([])

    assert result == [], f"Expected [], got {result!r}"


@pytest.mark.asyncio
async def test_encode_pages_empty_result_from_client_is_passed_through() -> None:
    """A non-empty input that returns an empty result from the client is passed through."""
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(return_value=[])
    encoder = _make_encoder(fake_client)

    result = await encoder.encode_pages([b"img1", b"img2"])

    assert result == [], f"Empty result from client must be returned unchanged, got {result!r}"


@pytest.mark.asyncio
async def test_encode_pages_success_returns_multi_vectors() -> None:
    """encode_pages success path returns the client's multi-vector list unchanged."""
    expected = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]]
    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(return_value=expected)
    encoder = _make_encoder(fake_client)

    result = await encoder.encode_pages([b"img1", b"img2"])

    assert result == expected, f"Expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# C. ColVisionEncoderConfig — served_name is exposed
# ---------------------------------------------------------------------------


def test_encoder_exposes_served_name_from_config() -> None:
    """ColVisionEncoder.served_name matches the served_name in the config."""
    cfg = _make_encoder_cfg()
    encoder = ColVisionEncoder(MagicMock(), cfg)

    assert encoder.served_name == "vidore/colqwen2.5-v0.2", (
        f"Expected served_name='vidore/colqwen2.5-v0.2', got {encoder.served_name!r}"
    )


# ---------------------------------------------------------------------------
# D. QdrantMultiVectorStore — ImportError path (qdrant_client not installed)
# ---------------------------------------------------------------------------


def test_qdrant_store_import_error_raises_visual_store_unavailable() -> None:
    """QdrantMultiVectorStore raises VisualStoreUnavailable when qdrant_client is missing."""
    # Hide qdrant_client in sys.modules to force the ImportError path regardless
    # of whether the package is installed in the test environment.
    real_module = sys.modules.pop("qdrant_client", None)
    try:
        with patch.dict(sys.modules, {"qdrant_client": None}):  # type: ignore[dict-item]
            from iw_rag.modes.visual.store import QdrantMultiVectorStore

            with pytest.raises(VisualStoreUnavailable, match="qdrant_client is not installed"):
                QdrantMultiVectorStore("http://localhost:6333")
    finally:
        if real_module is not None:
            sys.modules["qdrant_client"] = real_module


def test_qdrant_store_import_error_message_mentions_default_backend() -> None:
    """The ImportError message tells the operator to switch to postgres_candidates."""
    real_module = sys.modules.pop("qdrant_client", None)
    try:
        with patch.dict(sys.modules, {"qdrant_client": None}):  # type: ignore[dict-item]
            from iw_rag.modes.visual.store import QdrantMultiVectorStore

            with pytest.raises(VisualStoreUnavailable, match="postgres_candidates"):
                QdrantMultiVectorStore("http://localhost:6333")
    finally:
        if real_module is not None:
            sys.modules["qdrant_client"] = real_module


# ---------------------------------------------------------------------------
# E. QdrantMultiVectorStore — stub seam fires even when qdrant_client is available
# ---------------------------------------------------------------------------


def test_qdrant_store_stub_seam_raises_visual_store_unavailable_when_importable() -> None:
    """QdrantMultiVectorStore raises VisualStoreUnavailable even when qdrant_client is importable.

    The stub seam is unconditional (ADR-007): the Qdrant path is only wired
    at the T1 headline trigger.  If qdrant_client IS importable, the init
    proceeds past the ImportError guard but then immediately raises
    VisualStoreUnavailable (stub/seam message).
    """
    # Inject a fake qdrant_client module so the guarded import succeeds.
    fake_qdrant = ModuleType("qdrant_client")
    real_module = sys.modules.get("qdrant_client")
    sys.modules["qdrant_client"] = fake_qdrant
    try:
        from iw_rag.modes.visual.store import QdrantMultiVectorStore

        with pytest.raises(VisualStoreUnavailable, match="stub/seam"):
            QdrantMultiVectorStore("http://localhost:6333")
    finally:
        if real_module is not None:
            sys.modules["qdrant_client"] = real_module
        else:
            sys.modules.pop("qdrant_client", None)


def test_qdrant_store_stub_seam_message_mentions_adr007() -> None:
    """The stub/seam VisualStoreUnavailable message references ADR-007."""
    fake_qdrant = ModuleType("qdrant_client")
    real_module = sys.modules.get("qdrant_client")
    sys.modules["qdrant_client"] = fake_qdrant
    try:
        from iw_rag.modes.visual.store import QdrantMultiVectorStore

        with pytest.raises(VisualStoreUnavailable, match="ADR-007"):
            QdrantMultiVectorStore("http://localhost:6333")
    finally:
        if real_module is not None:
            sys.modules["qdrant_client"] = real_module
        else:
            sys.modules.pop("qdrant_client", None)


def test_qdrant_store_stub_seam_message_contains_provided_url() -> None:
    """The stub/seam error message includes the URL passed to __init__."""
    fake_qdrant = ModuleType("qdrant_client")
    real_module = sys.modules.get("qdrant_client")
    sys.modules["qdrant_client"] = fake_qdrant
    test_url = "http://my-qdrant-server:6333"
    try:
        from iw_rag.modes.visual.store import QdrantMultiVectorStore

        with pytest.raises(VisualStoreUnavailable, match=test_url):
            QdrantMultiVectorStore(test_url)
    finally:
        if real_module is not None:
            sys.modules["qdrant_client"] = real_module
        else:
            sys.modules.pop("qdrant_client", None)


# ---------------------------------------------------------------------------
# F. make_store factory — routing and ValueError
# ---------------------------------------------------------------------------


def test_make_store_postgres_candidates_returns_postgres_candidate_store() -> None:
    """make_store('postgres_candidates') returns a PostgresCandidateStore instance."""
    store = make_store("postgres_candidates", session_factory=MagicMock())

    assert type(store).__name__ == "PostgresCandidateStore", (
        f"Expected PostgresCandidateStore, got {type(store).__name__!r}"
    )


def test_make_store_postgres_candidates_wires_session_factory() -> None:
    """make_store('postgres_candidates') passes the session_factory to the store."""
    fake_factory = MagicMock()
    store = make_store("postgres_candidates", session_factory=fake_factory)

    assert isinstance(store, PostgresCandidateStore)
    # The session_factory is stored as _session_factory (internal attr)
    assert store._session_factory is fake_factory, (
        "session_factory must be wired into the store's _session_factory"
    )


def test_make_store_unknown_backend_raises_value_error() -> None:
    """make_store with an unknown backend name raises ValueError with match=."""
    with pytest.raises(ValueError, match="Unknown visual store backend"):
        make_store("unknown_backend")


def test_make_store_unknown_backend_message_contains_backend_name() -> None:
    """ValueError message from make_store includes the invalid backend string."""
    with pytest.raises(ValueError, match="bad_backend_name"):
        make_store("bad_backend_name")


def test_make_store_unknown_backend_message_lists_valid_backends() -> None:
    """ValueError message from make_store lists the valid backend strings."""
    with pytest.raises(ValueError, match="postgres_candidates"):
        make_store("not_a_backend")


def test_make_store_qdrant_backend_raises_visual_store_unavailable() -> None:
    """make_store('qdrant') propagates VisualStoreUnavailable from QdrantMultiVectorStore."""
    # Regardless of qdrant_client install status, the store raises VisualStoreUnavailable.
    with pytest.raises(VisualStoreUnavailable, match="qdrant"):
        make_store("qdrant", qdrant_url="http://localhost:6333")


def test_make_store_qdrant_backend_error_is_visual_store_unavailable_type() -> None:
    """make_store('qdrant') raises exactly VisualStoreUnavailable (not ValueError)."""
    exc: BaseException | None = None
    try:
        make_store("qdrant", qdrant_url="http://localhost:6333")
    except VisualStoreUnavailable as e:
        exc = e
    except Exception as e:
        exc = e

    assert type(exc).__name__ == "VisualStoreUnavailable", (
        f"Expected VisualStoreUnavailable, got {type(exc).__name__!r}: {exc}"
    )
