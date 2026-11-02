#!/usr/bin/env bash
# Deploy GreenMind AI to Google Cloud Run. Run this in Google Cloud Shell from this folder.
# Usage: bash deploy.sh <PROJECT_ID> [REGION]
set -euo pipefail
PROJECT_ID="${1:?Usage: bash deploy.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy greenmind \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 4Gi --cpu 2 \
  --timeout 300 --concurrency 4 \
  --min-instances 0 --max-instances 1 \
  --set-env-vars "SECRET_KEY=$(openssl rand -hex 24)"

echo
echo "Done. Your app URL is printed above (Service URL)."
