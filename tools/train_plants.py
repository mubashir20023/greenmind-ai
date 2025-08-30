#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a multi-species plant classifier from an ImageFolder dataset:
  data/plants/
    train/<Species_Name>/*.jpg
    val/<Species_Name>/*.jpg

Usage (fast baseline on CPU/GPU):
  python tools/train_plants.py --data_root data/plants --model resnet18 --epochs 12

Or (stronger, needs GPU for speed):
  python tools/train_plants.py --data_root data/plants --model vit_base_patch16_224 --epochs 20 --batch_size 32

Outputs:
  models/checkpoints/plant_id_<model>_best.pth
  models/checkpoints/plant_id_<model>_final.pth
  models/checkpoints/plant_id_<model>_resume.pth   # <-- auto-resume state
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np

def get_transforms(model_name: str):
    # Reasonable defaults for both ResNet and ViT
    train_t = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_t, val_t

def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res

def compute_class_weights(train_dataset):
    # ImageFolder provides .targets (list of class indices)
    counts = Counter(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        # inverse frequency weighting
        freq = counts.get(c, 1)
        weights.append(total / (freq * num_classes))
    w = torch.tensor(weights, dtype=torch.float32)
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/plants", help="Path containing train/ and val/")
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet18","vit_base_patch16_224"],
                    help="Backbone to fine-tune")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--output_dir", default="models/checkpoints")
    ap.add_argument("--amp", action="store_true", help="Force AMP; auto if CUDA available when not set")
    ap.add_argument("--no_resume", action="store_true", help="Disable auto-resume even if a resume file exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp or torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    assert train_dir.exists() and val_dir.exists(), f"Expect {train_dir} and {val_dir}"

    # Data
    train_t, val_t = get_transforms(args.model)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_t)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_t)

    num_classes = len(train_ds.classes)
    print(f"[info] Classes: {num_classes}")
    print(f"[info] Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    # Save the class map (so the app can decode indices to names)
    class_map_dir = Path("models/class_maps")
    class_map_dir.mkdir(parents=True, exist_ok=True)
    species_to_idx = {cls: i for i, cls in enumerate(train_ds.classes)}
    idx_to_species = {i: cls for cls, i in species_to_idx.items()}
    with (class_map_dir / "species_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(species_to_idx, f, ensure_ascii=False, indent=2)
    with (class_map_dir / "idx_to_species.json").open("w", encoding="utf-8") as f:
        json.dump(idx_to_species, f, ensure_ascii=False, indent=2)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Model
    if args.model == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model.to(device)

    # Loss (class weights + label smoothing)
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # AMP scaler (new API style)
    scaler = torch.amp.GradScaler(amp_device_type, enabled=use_amp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path   = out_dir / f"plant_id_{args.model}_best.pth"
    final_path  = out_dir / f"plant_id_{args.model}_final.pth"
    resume_path = out_dir / f"plant_id_{args.model}_resume.pth"

    # -------- Auto-resume --------
    start_epoch = 1
    if resume_path.exists() and not args.no_resume:
        try:
            ckpt = torch.load(resume_path, map_location="cpu")
            # Sanity checks
            if ckpt.get("model_name") == args.model and ckpt.get("num_classes") == num_classes:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and ckpt["scaler"] is not None:
                    scaler.load_state_dict(ckpt["scaler"])
                best_acc = float(ckpt.get("best_acc", 0.0))
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                print(f"[resume] Resumed from {resume_path} at epoch {start_epoch-1} (best@1={best_acc:.2f}%)")
            else:
                print("[resume] Resume file exists but is incompatible (model/num_classes changed). Starting fresh.")
        except Exception as e:
            print(f"[resume] Could not resume ({e}). Starting fresh.")

    # -------- Training loop --------
    for epoch in range(start_epoch, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        n_train = 0

        total_batches = len(train_loader)
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                top1 = topk_accuracy(logits, labels, topk=(1,))[0]
            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_top1 += top1 * bs
            n_train += bs

            # Progress log
            if (batch_idx % 50 == 0) or (batch_idx == total_batches):
                print(f"  [epoch {epoch}/{args.epochs}] "
                      f"batch {batch_idx}/{total_batches} "
                      f"loss={loss.item():.4f} acc={top1:.2f}%")

        scheduler.step()
        train_loss = running_loss / max(1, n_train)
        train_acc  = running_top1 / max(1, n_train)

        # ---- Validate ----
        model.eval()
        val_top1 = 0.0
        val_top5 = 0.0
        n_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                acc1, acc5 = topk_accuracy(logits, labels, topk=(1,5))
                bs = imgs.size(0)
                val_top1 += acc1 * bs
                val_top5 += acc5 * bs
                n_val += bs

        val_acc1 = val_top1 / max(1, n_val)
        val_acc5 = val_top5 / max(1, n_val)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} | train_acc {train_acc:.2f}% | "
              f"val@1 {val_acc1:.2f}% | val@5 {val_acc5:.2f}%")

        # Save best
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), best_path)
            print(f"[save] New best model â†’ {best_path} (val@1={best_acc:.2f}%)")

        # Save resume state each epoch
        torch.save({
            "epoch": epoch,
            "best_acc": best_acc,
            "model_name": args.model,
            "num_classes": num_classes,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
        }, resume_path)

    # Save final
    torch.save(model.state_dict(), final_path)
    print(f"[done] Best val@1: {best_acc:.2f}%")
    print(f"[paths] Best:   {best_path}")
    print(f"[paths] Final:  {final_path}")
    print(f"[paths] Resume: {resume_path}")

if __name__ == "__main__":
    # Windows friendliness
    torch.multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

