# Stage 1: Build & Train
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-train.txt .
RUN pip install --no-cache-dir --user -r requirements-train.txt

COPY training/ training/
COPY data/ data/
COPY params.yaml .

ARG MODEL_VERSION=1.0.0
RUN python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version ${MODEL_VERSION}


# Stage 2: Runtime Serving
FROM python:3.11-slim AS runner

ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      maintainer="Huzaifa (23F-0006)"

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    PORT=8000

WORKDIR /app

# Install curl for healthcheck & clean up apt lists
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned serving dependencies before copying application code for layer caching
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy model artifacts from builder stage (no dataset, no training scripts)
COPY --from=builder /build/artifacts/ artifacts/

# Copy application code
COPY app/ app/

# Non-root user setup
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
