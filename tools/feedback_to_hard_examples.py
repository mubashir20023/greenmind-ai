#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert user feedback into an ImageFolder tree for fine-tuning:

Input structure (created by /feedback API):
  data/feedback/
    2025-08-24/
      <uuid>/
        image.jpg
        meta.json       # { verdict, true_label, notes, model, best, alternatives, ... }

This script finds all feedback entries where:
  - verdict == "wrong"
  - true_label is provided (non-empty)

Then copies images to:
  data/hard_examples/train/<sanitized_true_label>/<uuid>.jpg

Usage:
  python tools/feedback_to_hard_examples.py
  python tools/feedback_to_hard_examples.py --feedback_root data/feedback --out_dir data/hard_examples

Options:
  --move          Move files instead of copy (default: copy)
  --val_split 0.0 Put a fraction into val/ instead of train/ (default: 0.0)
  --limit N       Limit max number of items processed (debug)
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import re
import random

IMAGE_CANDIDATES = ("image.jpg", "image.jpeg", "image.png", "image.webp")

def sanitize_label(label: str) -> str:
    """
    Convert a free-text label into a safe folder name:
    - lowercases
    - replaces spaces and slashes with underscores
    - keeps letters, numbers, underscores, hyphens
    """
    s = label.strip().lower()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    s = s.strip("_")
    return s or "unknown_label"

def find_image(dirpath: Path) -> Path | None:
    for name in IMAGE_CANDIDATES:
        p = dirpath / name
        if p.exists():
            return p
    # fallback: first file with typical image extensions
    for p in dirpath.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            return p
    return None

def collect_feedback_dirs(feedback_root: Path):
    """
    Yield paths like: data/feedback/YYYY-MM-DD/<uuid>
    Must contain meta.json.
    """
    if not feedback_root.exists():
        return
    for day_dir in sorted(feedback_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for case_dir in sorted(day_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if (case_dir / "meta.json").exists():
                yield case_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback_root", default="data/feedback", help="Root folder saved by /feedback")
    ap.add_argument("--out_dir", default="data/hard_examples", help="Output ImageFolder root")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")
    ap.add_argument("--val_split", type=float, default=0.0, help="Fraction to send to val/ (0..1)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items processed (debug)")
    args = ap.parse_args()

    feedback_root = Path(args.feedback_root)
    out_root = Path(args.out_dir)
    out_train = out_root / "train"
    out_val = out_root / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # Slightly randomized distribution for val_split
    rng = random.Random(1337)

    processed = 0
    skipped_no_true = 0
    skipped_no_img = 0
    total_seen = 0

    for case_dir in collect_feedback_dirs(feedback_root):
        total_seen += 1
        meta_path = case_dir / "meta.json"
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[warn] Cannot read meta.json in {case_dir}: {e}", file=sys.stderr)
            continue

        if meta.get("verdict") != "wrong":
            continue

        true_label = (meta.get("true_label") or "").strip()
        if not true_label:
            skipped_no_true += 1
            continue

        img_path = find_image(case_dir)
        if not img_path:
            skipped_no_img += 1
            continue

        cls = sanitize_label(true_label)
        # Decide split
        use_val = (args.val_split > 0.0) and (rng.random() < args.val_split)
        out_split_dir = (out_val if use_val else out_train) / cls
        out_split_dir.mkdir(parents=True, exist_ok=True)

        # Ensure unique file name
        dest_name = f"{case_dir.name}{img_path.suffix.lower()}"
        dest_path = out_split_dir / dest_name

        # Copy or move
        try:
            if args.move:
                shutil.move(str(img_path), str(dest_path))
            else:
                shutil.copy2(str(img_path), str(dest_path))
            processed += 1
        except Exception as e:
            print(f"[warn] Failed to write {dest_path}: {e}", file=sys.stderr)

        if args.limit and processed >= args.limit:
            break

    print("\n[feedback_to_hard_examples] Summary")
    print(f"  feedback root   : {feedback_root}")
    print(f"  out dir         : {out_root}")
    print(f"  total cases seen: {total_seen}")
    print(f"  processed       : {processed}")
    print(f"  skipped (no true_label): {skipped_no_true}")
    print(f"  skipped (no image)     : {skipped_no_img}")
    if args.val_split > 0:
        print(f"  val split       : {args.val_split:.2f}")
    print("Done.")

if __name__ == "__main__":
    main()
