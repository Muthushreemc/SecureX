# ── Stage 1: Build wheels ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels requirements.txt
COPY app.py consumer.py producer.py train_models.py dashboard.py ./
RUN mkdir -p /app/models && chown -R appuser:appgroup /app
LABEL org.opencontainers.image.title="Fraud Detection System" \
      org.opencontainers.image.version="1.0.0"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    WORKERS=3 \
    WORKER_TIMEOUT=30 \
    LOG_LEVEL=info
EXPOSE 5000
USER appuser
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:5000/health',timeout=4); sys.exit(0)"
CMD gunicorn app:app \
      --bind "0.0.0.0:${PORT}" \
      --workers "${WORKERS}" \
      --worker-class sync \
      --timeout "${WORKER_TIMEOUT}" \
      --log-level "${LOG_LEVEL}" \
      --access-logfile - \
      --error-logfile - \
      --capture-output
