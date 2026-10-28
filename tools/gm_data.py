# -*- coding: utf-8 -*-
"""Shared data helpers for train_classifier.py and evaluate.py."""

import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

NORMS = {
    # species model (app/identify.py) uses ImageNet statistics
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # disease model (app/health.py) scales to [-1, 1]
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def build_transforms(img_size: int, norm: str, train: bool, resize_mode: str = "crop"):
    mean, std = NORMS[norm]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if resize_mode == "square":  # matches app/health.py (cv2 resize to s x s)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([  # matches app/identify.py
        transforms.Resize(int(round(img_size * 256 / 224))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def read_manifest(path, split=None, root=None):
    """Return list of (abs_path, label). Relative paths resolve against `root`
    (default: the manifest's folder)."""
    path = Path(path)
    base = Path(root) if root else path.parent
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            p = Path(r["path"])
            if not p.is_absolute():
                p = base / p
            rows.append((str(p), r["label"]))
    return rows


def read_imagefolder(root):
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in exts:
                rows.append((str(p), cls_dir.name))
    return rows


def load_classes(path):
    """Accept a list (["a","b"]) or an idx->name dict ({"0":"a"})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data[str(i)] if str(i) in data else data[i] for i in range(len(data))]
    return list(data)


class ManifestDataset(Dataset):
    def __init__(self, rows, classes, transform):
        idx = {c: i for i, c in enumerate(classes)}
        self.items = [(p, idx[l]) for p, l in rows if l in idx]
        self.n_unknown = len(rows) - len(self.items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.transform(im)
        return x, y

    @property
    def targets(self):
        return [y for _, y in self.items]


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
