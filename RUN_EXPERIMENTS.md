# GreenMind AI: experiments for the paper

All experiments run on **Kaggle** (free). Nothing needs to be stored on your laptop.
Every number in the paper comes from the `metrics.json` files these notebooks produce.

| Notebook | What it produces | Kaggle settings | Time |
|---|---|---|---|
| `notebooks/01_build_dataset_and_test_deployed_model.ipynb` | iNaturalist dataset for the 1,106 species (pre-/post-Sept-2025 pools) + accuracy of the **deployed** ResNet-18 on photos it never saw | CPU, Internet On | 3–5 h |
| `notebooks/02_train_species_models.ipynb` | 139k-image balanced split, retrained ResNet-18 and EfficientNet-B0, in-distribution + temporal test results | GPU, Internet On, input = Notebook 1 output | 5–8 h |
| `notebooks/03_disease_lab_to_field.ipynb` | Disease models on PlantVillage, tested on PlantDoc field photos (lab-to-field gap) | GPU, Internet On, inputs = PlantVillage + PlantDoc | 1–2 h |

Notebooks 1 and 3 can run at the same time.

## Step 0: push the scripts to GitHub (once, on your laptop)

In the Anaconda prompt:

```
cd G:\projects\greenmind-ai
git add -A
git commit -m "Add reproducible experiment pipeline (iNaturalist fetch, split, train, evaluate, Kaggle notebooks)"
git push
```

The notebooks clone the repo from GitHub, so they only see the new scripts after this push.

## Step 1: Kaggle account (once)

1. Sign in at kaggle.com, open **Settings**, then **Phone verification** (required for Internet and GPU).

## Step 2: run a notebook

For each notebook:

1. kaggle.com, then **Create**, then **New Notebook**, then **File → Import Notebook**, and upload the `.ipynb` from `G:\projects\greenmind-ai\notebooks\`.
2. In the right panel, set **Session options**:
   - Accelerator: *None* for Notebook 1, *GPU T4 x2* for Notebooks 2 and 3.
   - Internet: **On**.
3. Add the inputs listed in the notebook's first cell (**Add Input** button).
4. Click **Save Version**, then **Save & Run All (Commit)**, then **Save**. It runs in the background; you can close the tab.
5. When it finishes (the bell icon notifies you), open the version and go to the **Output** tab.

Notebook 2's input is Notebook 1: **Add Input → Your Work → Notebooks**, then pick Notebook 1.

## Step 3: send me the results

From each finished notebook's **Output** tab, download the `results` folder and any `eval_test` folders,
plus `fetch_stats.json` and `split_stats.json`. You can also paste the text the last cell prints.
I'll turn them into the paper's tables and figures.

## Optional: your own Karachi photos (strong extra contribution)

Put 100–300 of your own photos in `G:\data\karachi\<Species_name>\` (folder names exactly as in
`models\class_maps\idx_to_species.json`), then run in the Anaconda prompt:

```
python tools\evaluate.py --ckpt models\checkpoints\plant_id_resnet18_best.pth --arch resnet18 ^
    --classes models\class_maps\idx_to_species.json --imagefolder G:\data\karachi --out results\e5_species_karachi
```

## Scripts

| Script | Purpose |
|---|---|
| `tools/inat_fetch_species.py` | Download research-grade, CC-licensed iNaturalist photos per species, one photo per observation, split by upload date |
| `tools/make_split.py` | Balanced, seeded, deduplicated train/val/test split; writes `manifest.csv` and `split_stats.json` |
| `tools/train_classifier.py` | Train any timm model from a manifest (AMP, cosine LR, early stopping, auto-resume) |
| `tools/evaluate.py` | Top-1/top-5, macro/weighted F1, balanced accuracy, ECE, bootstrap CI, CPU latency, confusion matrix |
