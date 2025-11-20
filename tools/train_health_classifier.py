# tools/train_health_classifier.py
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]  # PLANT-FYP root
DATA_DIR = ROOT / "data" / "plant_diseases" / "plantvillage_kaggle"
OUT_DIR = ROOT / "models" / "health"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = OUT_DIR / "leaf_health_effb3.pt"
CLASSES_PATH = OUT_DIR / "classes.json"

# ---------- Hyperparameters (lighter for CPU laptop) ----------
BATCH_SIZE = 8          # smaller batch to reduce RAM/compute
IMG_SIZE = 224          # smaller images than 384
EPOCHS = 3              # start small, you can increase later
LR = 1e-4
VAL_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0         # IMPORTANT on Windows to avoid dataloader hangs
PRINT_EVERY = 50        # batches between progress prints


def main():
    print(f"Using device: {DEVICE}")
    print(f"Dataset dir: {DATA_DIR}")

    # ---------- Transforms ----------
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---------- Dataset ----------
    print("Loading dataset with ImageFolder...")
    full_ds = datasets.ImageFolder(str(DATA_DIR), transform=train_tfms)
    num_classes = len(full_ds.classes)
    print("Found", len(full_ds), "images in", num_classes, "classes")
    print("Classes example:", full_ds.classes[:10])

    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # apply val transforms to underlying dataset
    val_ds.dataset.transform = val_tfms

    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---------- Model ----------
    print("Loading EfficientNet-B0 backbone...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)
    print("Model ready.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        print(f"\nEpoch {epoch}/{EPOCHS} - training...")
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % PRINT_EVERY == 0 or batch_idx == 1:
                print(
                    f"  [train] epoch {epoch}, batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validate ----
        print(f"Epoch {epoch}/{EPOCHS} - validating...")
        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}"
        )

        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, WEIGHTS_PATH)
            print(f"  -> saved best model to {WEIGHTS_PATH}")

    # Save class names so app.health can map indices -> labels
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(full_ds.classes, f, ensure_ascii=False, indent=2)
    print("Saved class list to", CLASSES_PATH)
    print("Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
