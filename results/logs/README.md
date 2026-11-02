# Experiment logs

Raw Kaggle logs of the runs reported in the paper and the README.

| Log | Notebook | Content |
|---|---|---|
| `01_build_dataset_first_run.log` | 01 | First iNaturalist download (1,084 species resolved); its evaluation step ran out of memory, fixed in commit 018d43f |
| `01b_dataset_download_inat_tar.log` | 02 (download mode) | Second download, saved as `inat.tar` (Pool A 138,463 / Pool B 19,303 images) |
| `02_train_species_models.log` | 02 | Original model on Pool B; 958-species split; ResNet-18 and EfficientNet-B0 training and evaluation |
| `03_disease_lab_to_field.log` | 03 | PlantVillage split, EfficientNet-B0 and ResNet-18 (seed 42), PlantDoc field test |
| `04_disease_improvements.log` | 04 | Seeds 1 and 2, field-data training, leaf crops (original detector), calibration, abstention, occlusion |
| `05_leaf_detector.log` | 05 | YOLOv8s leaf detector on PlantDoc boxes (mAP@0.5 0.932), leaf-crop test, app model bundle |
