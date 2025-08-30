#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a trained YOLO detector over your dataset to produce flower crops for classifier training.

Usage:
  python tools/make_flower_crops.py --in_root data/flowers --out_root data/flowers_crops \
      --weights runs/detect/train/weights/best.pt --class_name flower
"""

import argparse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import hashlib
import shutil

def stable_name(p: Path, idx: int) -> str:
    h = hashlib.sha1((str(p) + f"#{idx}").encode("utf-8")).hexdigest()[:16]
    return f"{h}.jpg"

def largest_box(boxes):
    # boxes: list of (x1,y1,x2,y2,conf)
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    x1 = (w - s) // 2
    y1 = (h - s) // 2
    return img.crop((x1, y1, x1 + s, y1 + s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="ImageFolder root with train/ and val/")
    ap.add_argument("--out_root", required=True, help="Output ImageFolder root")
    ap.add_argument("--weights", required=True, help="YOLO weights .pt")
    ap.add_argument("--class_name", default="flower", help="Detector class to keep")
    ap.add_argument("--min_crop_size", type=int, default=96, help="min side in pixels")
    ap.add_argument("--fallback_center", action="store_true", help="Use center square if no detection")
    args = ap.parse_args()

    in_root  = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)

    for split in ["train", "val"]:
        in_split = in_root / split
        if not in_split.exists():
            continue
        for cls_dir in sorted(in_split.iterdir()):
            if not cls_dir.is_dir():
                continue
            rel = cls_dir.relative_to(in_root)
            out_cls = out_root / rel
            out_cls.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                res = yolo.predict(source=img, conf=0.25, max_det=30, verbose=False)
                dets = res[0]
                names = dets.names

                boxes = []
                if dets.boxes is not None and len(dets.boxes) > 0:
                    for b in dets.boxes:
                        cls_id = int(b.cls.item())
                        cls_name = str(names.get(cls_id, cls_id)).lower()
                        if cls_name != args.class_name.lower():
                            continue
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        # filter tiny crops
                        if (x2 - x1) < args.min_crop_size or (y2 - y1) < args.min_crop_size:
                            continue
                        conf = float(b.conf.item())
                        boxes.append((x1, y1, x2, y2, conf))

                if boxes:
                    x1, y1, x2, y2, _ = largest_box(boxes)
                    crop = img.crop((x1, y1, x2, y2))
                elif args.fallback_center:
                    crop = center_square(img)
                else:
                    # copy original if no detection (keeps sample count)
                    out_path = out_cls / stable_name(img_path, 0)
                    shutil.copy2(img_path, out_path)
                    continue

                out_path = out_cls / stable_name(img_path, 0)
                crop.save(out_path, quality=95)

            print(f"[done] {out_cls}")

    print(f"[ok] Crops at: {out_root}")

if __name__ == "__main__":
    main()
