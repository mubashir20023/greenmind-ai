# Reproducing the GreenMind experiments

Everything runs on **Kaggle** (free: 30 GPU hours per week, phone verification required).
Every number in the paper and the README comes from the `metrics.json` / JSON files these notebooks write;
the raw logs of our runs are in [`results/logs/`](results/logs).

## General procedure (for every notebook)

1. kaggle.com → **Create → New Notebook → File → Import Notebook** → choose the `.ipynb` from [`notebooks/`](notebooks).
2. Right panel → **Session options**: set the **Accelerator** listed below and **Internet: On**.
3. **Add Input** → *Notebook* chip → add the inputs listed below (a green tick appears).
4. Run the **first cells by hand** (▶) until the data paths print. This is needed once per notebook:
   Kaggle only lets a *committed* run use datasets that are already attached.
5. **Save Version → Save & Run All (Commit) → Save**, then stop the draft session (⏻) so it does not use GPU time.
6. When the version shows *Successfully ran*, open its **Logs** tab → *Download Logs*, and its **Output** tab for files.

Never edit a notebook whose output another notebook uses: the newest version's output is the one that gets attached.

## The notebooks

| # | Notebook | Accelerator | Inputs | Time | Produces |
|---|---|---|---|---|---|
| 1 | `01_build_dataset_and_test_deployed_model` | None (CPU) | – | ~6 h | `inat.tar` (Pool A + Pool B), `fetch_stats.json` |
| 2 | `02_train_species_models` | GPU T4 x2 | notebook 1 *(optional; without it the images are downloaded first, +6 h)* | ~4.5 h | 958-species split, ResNet-18 + EfficientNet-B0, temporal and original-model results |
| 3 | `03_disease_lab_to_field` | GPU T4 x2 | – (PlantVillage is attached in step 4) | ~1.2 h | PlantVillage-trained disease models, PlantDoc field results |
| 4 | `04_disease_improvements` | GPU T4 x2 | – (PlantVillage is attached in step 4) | ~2.6 h | seeds, field-data training, calibration, abstention, occlusion test |
| 5 | `05_leaf_detector` | GPU T4 x2 | notebooks 2 and 4 | ~1.5 h | YOLO leaf detector, leaf-crop test, `app_models/` bundle |

Notebooks 1, 3 and 4 are independent and can run at the same time (Kaggle allows a few concurrent sessions).

## Putting the new models into the app

Download `app_models/` from notebook 5's Output tab and copy the files into the project:

| File | Destination |
|---|---|
| `plant_id_effb0_best.pth` | `models/checkpoints/plant_id_effb0_best.pth` |
| `idx_to_species_effb0.json` | `models/class_maps/idx_to_species_effb0.json` |
| `leaf_health_effb0_field.pth` | `models/health/leaf_health_effb0_field.pth` |
| `classes.json` | `models/health/classes.json` |
| `yolo_leaf.pt` | `models/health/yolo_leaf.pt` |

The app picks the new files up automatically (see `.env.example`). Commit them with Git LFS:

```
git lfs install
git add models/ && git commit -m "Add retrained models" && git push
```

## Scripts

| Script | Purpose |
|---|---|
| `tools/inat_fetch_species.py` | Research-grade, CC-licensed iNaturalist photos per species, one per observation, split by upload date; records licence and attribution |
| `tools/make_split.py` | Balanced, seeded, MD5-de-duplicated train/val/test split → `manifest.csv`, `split_stats.json` |
| `tools/train_classifier.py` | Any timm model from a manifest (AMP, AdamW, warm-up + cosine, label smoothing, early stopping, auto-resume) |
| `tools/evaluate.py` | Top-1/top-5, macro/weighted F1, balanced accuracy, ECE, bootstrap CI, CPU latency, confusion matrix, per-image predictions |
| `tools/disease_extras.py` | Field-data manifest, YOLO leaf crops, temperature scaling + risk-coverage, occlusion test |
| `tools/plantdoc_leaf_yolo.py` | PlantDoc detection boxes (VOC) → single-class YOLO dataset, excluding test-image duplicates |

## Running locally (optional)

The evaluation script also runs on a CPU, for example on your own photographs
(one sub-folder per class, folder names as in the class map):

```
python tools/evaluate.py --ckpt models/checkpoints/plant_id_effb0_best.pth --arch efficientnet_b0 ^
    --classes models/class_maps/idx_to_species_effb0.json --imagefolder G:\data\karachi --out results\karachi
```
