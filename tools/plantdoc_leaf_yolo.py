#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert the PlantDoc object-detection dataset (Pascal VOC XML, github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)
into a single-class YOLO dataset ("leaf"), for training a leaf detector for GreenMind's hybrid disease mode.

* TRAIN is split into train/val (default 90/10); TEST is kept as the detection test split.
* Any TRAIN image that is byte-identical (MD5) to an image in --exclude_dir (e.g. the PlantDoc
  classification test split used to evaluate the disease classifier) is dropped, so the detector
  never sees an evaluation image.
* Files are copied under simple numbered names (PlantDoc file names contain '%', '?', '+').

Usage:
  python tools/plantdoc_leaf_yolo.py --od_root PlantDoc-Object-Detection-Dataset \
      --exclude_dir PlantDoc-Dataset/test --out /kaggle/working/leafdet
"""

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_image(xml_path, root):
    stem = xml_path.with_suffix("")
    for ext in IMG_EXT:
        c = Path(str(stem) + ext)
        if c.exists():
            return c
    fn = (root.findtext("filename") or "").strip()
    if fn and (xml_path.parent / fn).exists():
        return xml_path.parent / fn
    return None


def parse(xml_path):
    root = ET.parse(xml_path).getroot()
    img = find_image(xml_path, root)
    if img is None:
        return None, []
    try:
        with Image.open(img) as im:
            W, H = im.size
    except Exception:
        return None, []
    boxes = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--od_root", required=True)
    ap.add_argument("--exclude_dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    od, out = Path(a.od_root), Path(a.out)
    excl = set()
    if a.exclude_dir:
        for p in Path(a.exclude_dir).rglob("*"):
            if p.is_file() and p.suffix in IMG_EXT:
                excl.add(md5(p))

    items = {"TRAIN": [], "TEST": []}
    dropped = {"no_image_or_boxes": 0, "duplicate_of_excluded": 0}
    for split in items:
        for x in sorted((od / split).glob("*.xml")):
            img, boxes = parse(x)
            if img is None or not boxes:
                dropped["no_image_or_boxes"] += 1
                continue
            if split == "TRAIN" and md5(img) in excl:
                dropped["duplicate_of_excluded"] += 1
                continue
            items[split].append((img, boxes))

    rng = random.Random(a.seed)
    rng.shuffle(items["TRAIN"])
    k = int(round(len(items["TRAIN"]) * a.val_frac))
    splits = {"val": items["TRAIN"][:k], "train": items["TRAIN"][k:], "test": items["TEST"]}

    stats = {"dropped": dropped}
    for name, rows in splits.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        nb = 0
        for i, (img, boxes) in enumerate(rows):
            dst = out / "images" / name / f"{name}_{i:05d}.jpg"
            with Image.open(img) as im:
                im.convert("RGB").save(dst, "JPEG", quality=95)
            with (out / "labels" / name / f"{name}_{i:05d}.txt").open("w") as f:
                for cx, cy, w, h in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            nb += len(boxes)
        stats[name] = {"images": len(rows), "boxes": nb}

    (out / "leaf.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n",
        encoding="utf-8")
    (out / "leafdet_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
