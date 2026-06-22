# iw-rag E2E api image — same runtime as docker/api.Dockerfile but also bakes
# `alembic/`, `alembic.ini` and `scripts/` so the container can migrate + seed
# itself, and serves as the image for the `app`, `e2e-models` (stub) services
# in docker/compose.e2e.yaml. Hot-path only — no GPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY scripts ./scripts
COPY docker/e2e-entrypoint.sh /usr/local/bin/e2e-entrypoint.sh
RUN chmod +x /usr/local/bin/e2e-entrypoint.sh \
    && uv sync --no-dev --frozen --group observability --group docman

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/e2e-entrypoint.sh"]
