# GreenMind AI — Smart plant recognition

Identify plants from photos (upload or live camera) and get **explainable** results with cited facts.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/web-Flask-lightgrey.svg)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-red.svg)

## Table of Contents
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Dataset & Training](#dataset--training)
- [Explainability](#explainability)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Limitations & Ethics](#limitations--ethics)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Highlights
- **Top-k** predictions with confidence bar and alternatives  
- **Grad-CAM / EigenCAM** overlays for model explanations  
- **Flask API**: `/identify`, `/explain`, `/feedback`  
- **Robust dataset pipeline** (iNaturalist → CSV → ImageFolder)  
- Modern, responsive UI: **upload / drag-drop / live camera**, feedback loop

## Architecture
lua
Copy code
               +------------------+
User (web/mobile) -> Web UI (HTML/CSS/JS)
+------------------+
|
v
Flask Backend
+------------------+
| /identify |
| /explain |
| /feedback |
+------------------+
| |
PyTorch Grad-CAM
Classifier (pytorch-grad-cam)
|
Model ckpt
|
+-------------------+
| Dataset Builder |
| CSV -> ImageFolder|
+-------------------+
^
iNaturalist (licensed)

shell
Copy code

## Quick Start

> Windows PowerShell shown; on macOS/Linux, drop the `^` line continuations.

1) **Create and activate a virtual env**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Configure environment
Copy .env.example to .env and fill paths/options as needed (no secrets in repo):

bash
Copy code
# Windows
copy .env.example .env
# macOS/Linux
# cp .env.example .env
Run the app (development)

bash
Copy code
# Option A: module entry
python -m app.main

# Option B: Flask runner
# set FLASK_APP=app.main:app && flask run
Production (Linux/macOS)

bash
Copy code
# Example: 4 workers on port 5000
gunicorn -w 4 -b 0.0.0.0:5000 app.main:app
Configuration
Edit .env (sample keys below—placeholders only):

ini
Copy code
# ---- Flask ----
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000

# ---- Models / Classifier ----
MODEL_NAME=resnet18
CKPT_PATH=models/checkpoints/plant_id_resnet18_best.pth
IDX2_PATH=models/class_maps/idx_to_species.json

# ---- YOLO Detector (optional) ----
YOLO_WEIGHTS=runs/detect/train/weights/best.pt
YOLO_CONF=0.25
YOLO_MAX_CROPS=5
TOPK=3
API Endpoints
POST /identify
Body: multipart/form-data with field file (image)
Response (JSON):

json
Copy code
{
  "model": "resnet18",
  "best": { "label": "Mangifera indica", "score": 0.92 },
  "alternatives": [
    {"label":"Mangifera odorata","score":0.04},
    {"label":"Spondias mombin","score":0.02}
  ],
  "facts_html": "<p>Common name: Mango...</p>"
}
cURL

bash
Copy code
curl -F "file=@sample.jpg" http://localhost:5000/identify
POST /explain?method=gradcam|eigen
Body: multipart/form-data with field file
Response: PNG heatmap (transparent) aligned to the input image.

POST /feedback
Body: multipart/form-data

file: the same image

meta: JSON blob

json
Copy code
{
  "verdict": "correct",
  "true_label": null,
  "notes": null,
  "model": "resnet18",
  "best": {"label":"...","score":0.92},
  "alternatives": [],
  "time": "2025-01-01T12:34:56Z"
}
Response: { "ok": true }

Dataset & Training
1) Build CSV from iNaturalist (resumable)
bash
Copy code
python tools/inat_build_flowers_csv.py ^
  --out_csv raw_data/flowers_1000.csv ^
  --num_species 1000 ^
  --min_per_class 80 ^
  --max_per_class 400 ^
  --resume ^
  --species_cache raw_data/flowers_species_cache.json
2) CSV → ImageFolder
bash
Copy code
python tools/inat_csv_to_imagefolder.py ^
  --csv raw_data/flowers_1000.csv ^
  --out_dir data/flowers ^
  --min_per_class 80 ^
  --max_per_class 400 ^
  --train_split 0.85 ^
  --workers 8
Result:

swift
Copy code
data/flowers/
  train/<species_slug>/*.jpg
  val/<species_slug>/*.jpg
  metadata/manifest.csv
3) Train
Use tools/train_plants.py (if included) or your own TIMM/ViT script.
Typical recipe: RandomResizedCrop, flips, RandAugment; AdamW + Cosine LR; report top-1/top-5 & per-class accuracy, confusion matrix.

Explainability
pytorch-grad-cam for Grad-CAM/EigenCAM

UI includes opacity slider and Save for the explanation PNG

Improves debugging and user trust

Screenshots
Add real screenshots for reviewers:

pgsql
Copy code
docs/
  screenshot-hero.png
  screenshot-identify.png
  screenshot-explain.png
In Markdown:

markdown
Copy code
![Identify](docs/screenshot-identify.png)
![Explainability](docs/screenshot-explain.png)
Project Structure
bash
Copy code
app/
  __init__.py
  main.py            # Flask entry
  identify.py        # /identify endpoint
  explain.py         # /explain endpoint (Grad-CAM/EigenCAM)
  facts.py           # facts HTML builder
  detector.py        # optional YOLO pipeline
  health.py          # health checks
models/
  class_maps/        # idx_to_species.json, species_to_idx.json
  checkpoints/       # (ignored or tracked with LFS)
tools/
  inat_build_flowers_csv.py
  inat_csv_to_imagefolder.py
  train_plants.py
web/
  templates/index.html
  static/style.css
  static/script.js
Limitations & Ethics
Species look-alikes and seasonality can cause confusion

Low-quality images (blur, partial organs) reduce accuracy

Safety: suggestions only — do not ingest plants based solely on the app

Respect image licenses (iNaturalist provides license codes; include attributions where required)

License
MIT — see LICENSE.

Acknowledgements
iNaturalist for images and taxonomy (respect photo licenses)

Open-source libraries: PyTorch, torchvision, timm, pytorch-grad-cam, Flask, Ultralytics

pgsql
Copy code

After pasting, hit **Preview** — the ASCII diagram should be inside a box, and there should be **no** “Copy code” words anywhere.
