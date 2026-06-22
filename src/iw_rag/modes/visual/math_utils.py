"""Pure math helpers for the visual late-interaction path (WI-031 / Task 4b).

All functions are deterministic, pure (no I/O), and unit-tested.

Two-clocks rule: these helpers are used by both the offline indexer and the
online retriever.  The indexer (pool_to_n, binary_quantize) is offline-only;
the retriever (maxsim, mean_pool) is hot-path-allowed.

References:
- task-4b-visual-rag-route §4 (pooling, quantization)
- task-4b-visual-rag-route §6 (two-stage MaxSim)
- ADR-010 (pooled+binary, no raw multi-vectors)
- R-00107 F3 (pooled_vectors_per_page≈38 + binary quantization)
"""

from __future__ import annotations

import math


def pool_to_n(
    vectors: list[list[float]],
    n: int,
) -> list[list[float]]:
    """Pool a tall matrix of patch vectors down to at most ``n`` rows via mean-pooling.

    Divides the input rows into ``n`` contiguous groups and mean-pools each
    group into one representative row.  If ``len(vectors) <= n``, returns the
    input unchanged (no padding — downstream store tolerates variable row counts
    within the ``pooled_vectors_per_page`` ceiling).

    The canonical target is ``n ≈ 38`` (~32 row-pooled + 6 special-token vectors
    per R-00107/F3), yielding a 1–2 orders-of-magnitude storage saving over raw
    multi-vectors (ADR-010).

    Args:
        vectors: The patch-vector matrix as a list of rows (each row is a
            list of floats of length ``d``).
        n: Target number of output rows.  Must be ≥ 1.

    Returns:
        A list of ``min(len(vectors), n)`` mean-pooled rows of the same
        dimensionality as the input.

    Raises:
        ValueError: If ``n < 1`` or ``vectors`` is empty.
    """
    if n < 1:
        raise ValueError(f"pool_to_n: n must be >= 1, got {n}")
    if not vectors:
        raise ValueError("pool_to_n: vectors must be non-empty")

    num = len(vectors)
    if num <= n:
        # Input is already within budget — return as-is (no-op, no padding).
        return vectors

    dims = len(vectors[0])
    result: list[list[float]] = []

    # Divide rows into n contiguous groups; each group is mean-pooled.
    # Group sizes differ by at most 1 (ceiling/floor).
    for i in range(n):
        # Inclusive start and exclusive end indices for this group.
        start = (i * num) // n
        end = ((i + 1) * num) // n
        group = vectors[start:end]
        # Mean-pool the group along the row axis.
        group_len = len(group)
        pooled = [sum(row[d] for row in group) / group_len for d in range(dims)]
        result.append(pooled)

    return result


def binary_quantize(vectors: list[list[float]]) -> list[list[int]]:
    """Binary-quantize a matrix: each float → 1 if ≥ 0 else 0, packed per row.

    Each input row of ``d`` floats is mapped to ``ceil(d / 8)`` bytes (ints in
    0–255) with the bits packed MSB-first.  This is the sign-bit binary
    quantization scheme used by ColPali/ColQwen2.5 for storage efficiency
    (ADR-010, R-00107/F3).

    Args:
        vectors: Matrix of float rows (each row same length ``d``).

    Returns:
        A list of int lists, where each inner list has ``ceil(d / 8)`` ints
        (byte values 0–255).  Same number of rows as the input.

    Raises:
        ValueError: If ``vectors`` is empty.
    """
    if not vectors:
        raise ValueError("binary_quantize: vectors must be non-empty")
    result: list[list[int]] = []
    for row in vectors:
        n_bits = len(row)
        n_bytes = math.ceil(n_bits / 8)
        bytes_out: list[int] = []
        for b in range(n_bytes):
            byte_val = 0
            for bit_idx in range(8):
                dim_idx = b * 8 + bit_idx
                # MSB-first: bit 7 is the first dimension in this byte
                if dim_idx < n_bits and row[dim_idx] >= 0:
                    byte_val |= 1 << (7 - bit_idx)
            bytes_out.append(byte_val)
        result.append(bytes_out)
    return result


def dequantize_signs(quantized_row: list[int], n_dims: int) -> list[int]:
    """Unpack a binary-quantized row back to per-dimension sign bits (1 or 0).

    Inverse of the MSB-first packing in :func:`binary_quantize`.  Used to
    verify round-trip consistency in tests.

    Args:
        quantized_row: A list of byte-packed ints from :func:`binary_quantize`.
        n_dims: The original number of float dimensions (needed to trim padding
            bits in the final byte).

    Returns:
        A list of ``n_dims`` ints where 1 represents a non-negative original
        value and 0 represents a negative original value.
    """
    signs: list[int] = []
    for byte_val in quantized_row:
        for bit_idx in range(8):
            dim_idx = len(signs)
            if dim_idx >= n_dims:
                break
            # MSB-first: bit 7 corresponds to the first dim in this byte.
            signs.append(1 if (byte_val >> (7 - bit_idx)) & 1 else 0)
    return signs[:n_dims]


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of vectors.

    Used to create the single-vector proxy for stage-1 pgvector HNSW prefetch
    (``PostgresCandidateStore``).  The mean-pooled proxy is stored alongside
    the quantized multi-vectors so the two-stage retriever can run stage-1
    cheaply in Postgres.

    Args:
        vectors: A non-empty list of float lists, all of the same length ``d``.

    Returns:
        A single list of ``d`` floats (the element-wise mean).

    Raises:
        ValueError: If ``vectors`` is empty.
    """
    if not vectors:
        raise ValueError("mean_pool: vectors must be non-empty")
    dims = len(vectors[0])
    n = len(vectors)
    return [sum(row[d] for row in vectors) / n for d in range(dims)]


def maxsim(
    query_vecs: list[list[float]],
    patch_vecs: list[list[float]],
    *,
    return_heatmap: bool = False,
) -> float | tuple[float, list[list[float]]]:
    """Compute the MaxSim (late-interaction) score between query tokens and patches.

    MaxSim = sum over query tokens t of max over patch tokens p of dot(t, p).
    This is the core ColPali / ColQwen2.5 scoring function (ADR-010 §4).

    The function is deliberately implemented in pure Python (no NumPy) so it
    runs in the test environment without heavy dependencies.  The production
    hot-path may use NumPy for vectorisation — but the logic here is the
    authoritative reference.

    Args:
        query_vecs: Query token vectors — shape ``[n_query, d]``.
        patch_vecs: Page patch vectors — shape ``[n_patch, d]``.
        return_heatmap: When ``True``, also return the ``[n_query × n_patch]``
            similarity matrix for the optional MaxSim heatmap (interpretability
            hook, R-00107/F6).

    Returns:
        When ``return_heatmap=False`` (default): a single float score.
        When ``return_heatmap=True``: ``(score, heatmap)`` where ``heatmap``
        is a ``list[list[float]]`` of shape ``[n_query, n_patch]``.
    """
    n_query = len(query_vecs)

    # Build the full similarity matrix: sim[i][j] = dot(query[i], patch[j])
    sim_matrix: list[list[float]] = []
    for q in query_vecs:
        row: list[float] = []
        for p in patch_vecs:
            dot = sum(qi * pi for qi, pi in zip(q, p, strict=True))
            row.append(dot)
        sim_matrix.append(row)

    # MaxSim: for each query token, take the max over all patch tokens.
    score = sum(max(sim_matrix[i]) for i in range(n_query)) if n_query > 0 else 0.0

    if return_heatmap:
        return (score, sim_matrix)
    return score
