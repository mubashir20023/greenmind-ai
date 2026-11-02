# GreenMind AI on Hugging Face Spaces (Docker SDK, free "CPU basic": 2 vCPU, 16 GB RAM).
# This file is self-contained: it clones the app from GitHub (with Git LFS weights) at build time,
# so the Space only needs this Dockerfile and README.md. Use "Factory rebuild" in the Space
# settings to pick up new commits from GitHub.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && git lfs install --system

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

ARG REPO_URL=https://github.com/mubashir20023/greenmind-ai
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} . && git lfs pull && rm -rf .git

RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user gunicorn

# SQLite inside the container (data resets when the Space restarts; set DATABASE_URL to an external
# MySQL/Postgres URL in the Space secrets to keep accounts and saved plants permanently).
ENV DATABASE_URL=sqlite:////home/user/app/data/greenmind.db \
    PORT=7860 FLASK_DEBUG=0 FLASK_ENV=production \
    HEALTH_INPUT_SIZE=224
RUN mkdir -p data runs/health

EXPOSE 7860
# One worker keeps a single copy of the models in memory; threads serve concurrent requests.
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "300", "-b", "0.0.0.0:7860", "app.main:app"]
