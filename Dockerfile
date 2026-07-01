FROM qdrant/qdrant:v1.9.2 AS qdrant

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/app/.cache/huggingface" \
    PATH="/app/.venv/bin:$PATH"

ARG EMBEDDING_MODEL=allenai/specter2_base
ARG EMBEDDING_DOCUMENT_ADAPTER=allenai/specter2
ARG EMBEDDING_QUERY_ADAPTER=allenai/specter2_adhoc_query
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV EMBEDDING_DOCUMENT_ADAPTER=${EMBEDDING_DOCUMENT_ADAPTER}
ENV EMBEDDING_QUERY_ADAPTER=${EMBEDDING_QUERY_ADAPTER}

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/
COPY --from=qdrant /qdrant /qdrant

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY static ./static
COPY config ./config

RUN uv sync --frozen --no-dev
RUN python -c "import os; from adapters import AutoAdapterModel; from transformers import AutoTokenizer; model_name = os.environ['EMBEDDING_MODEL']; AutoTokenizer.from_pretrained(model_name); model = AutoAdapterModel.from_pretrained(model_name); model.load_adapter(os.environ['EMBEDDING_DOCUMENT_ADAPTER'], load_as='proximity'); model.load_adapter(os.environ['EMBEDDING_QUERY_ADAPTER'], load_as='adhoc_query')"

COPY docker ./docker
RUN chmod +x docker/fly-start.sh

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
