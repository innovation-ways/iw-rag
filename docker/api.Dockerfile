# iw-rag API image (FastAPI). Hot-path is CPU-only; the only GPU touch is the
# benchmark GpuSampler (read-only NVML telemetry on POST /benchmark/run), which
# needs nvidia-ml-py in-image AND a GPU reservation on the api container in the
# benchmark overlay — it degrades to a no-op otherwise. Uses uv for a
# reproducible install from the committed uv.lock.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# uv (pinned installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System libs for the document-management upload path: tesseract (in-image OCR
# for invoices/scans) + poppler (optional per-page OCR of scanned PDFs). CPU-only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps first (cache-friendly), then the package.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
# --group observability installs OpenTelemetry SDK + OTLP gRPC exporter so the
# loop's span() instrumentation actually exports to Phoenix (without it tracing
# silently degrades to a no-op).
# --group docman installs the online upload-path deps (pypdf + Pillow + pytesseract).
# --group gpu-telemetry installs nvidia-ml-py (pynvml) for the benchmark GpuSampler;
# pure-python, no CUDA in-image — libnvidia-ml.so is injected by the nvidia runtime
# when the api container has a GPU reservation (benchmark overlay only).
# --group eval-ragas installs the LEAN Ragas runtime (ragas + langchain +
# langchain-openai + datasets, NO deepeval) so POST /benchmark/run can score
# the reference-required metrics (context_precision/recall, factual_correctness,
# noise_sensitivity) via the local judge + BGE-M3 embedder. Lazy-imported in
# benchmark_composition; degrades to None if absent (F-00093 S05).
# --group pdf installs Playwright so POST-run reports can be exported to PDF via
# headless Chromium (GET /benchmark/reports/{id}/pdf). The browser binary + its
# system libraries are installed below.
RUN uv sync --no-dev --frozen --group observability --group docman --group gpu-telemetry --group eval-ragas --group pdf

# Chromium + its system libraries for the PDF export path (Playwright). Adds
# ~400 MB to the image; only used by the benchmark report export route.
RUN uv run playwright install --with-deps chromium

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "iw_rag.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
