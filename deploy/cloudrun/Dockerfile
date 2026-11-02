# GreenMind AI on Google Cloud Run (also works on any Docker host).
# Self-contained: clones the app from GitHub (with Git LFS model weights) at build time,
# so you can deploy from this folder alone:  gcloud run deploy greenmind --source .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && git lfs install --system

WORKDIR /app
ARG REPO_URL=https://github.com/mubashir20023/greenmind-ai
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} . && git lfs pull && rm -rf .git

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# SQLite inside the container: accounts and saved plants reset when a new instance starts.
# Set DATABASE_URL to an external MySQL/Postgres URL to keep them.
ENV DATABASE_URL=sqlite:////app/data/greenmind.db \
    FLASK_DEBUG=0 FLASK_ENV=production HEALTH_INPUT_SIZE=224 \
    PYTHONUNBUFFERED=1 HOME=/app
RUN mkdir -p data runs/health

# Cloud Run sets $PORT (8080); one worker keeps one copy of the models in memory.
CMD exec gunicorn -w 1 --threads 4 --timeout 300 -b 0.0.0.0:${PORT:-8080} app.main:app
