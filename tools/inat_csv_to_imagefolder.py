#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iNaturalist CSV -> ImageFolder downloader

Reads an iNaturalist export CSV with columns including `image_url`, `scientific_name`
and downloads images into:

out_root/
  train/
    Genus_species/
      <files>.jpg
  val/
    Genus_species/
      <files>.jpg

Features:
- Cleans species folder names (safe snake_case)
- Skips rows without image_url
- Deduplicates by image_url
- Per-class min / max caps (keep classes with >= min; cap at max)
- Train/val split (stratified by simple slicing)
- Retries, timeouts, and partial failure handling
- Writes class maps: species_to_idx.json and idx_to_species.json
- Writes attribution.csv (image_url, saved_path, species)

Usage:
  python tools/inat_csv_to_imagefolder.py --csv inat_export.csv --out_dir data/plants \
         --min_per_class 30 --max_per_class 400 --train_split 0.8 --workers 8
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests


def sanitize_class_name(name: str) -> str:
    """Turn 'Mangifera indica' -> 'Mangifera_indica' (safe for Windows paths)."""
    name = name.strip()
    if not name:
        return ""
    # collapse whitespace to single space
    name = re.sub(r"\s+", " ", name)
    # replace spaces and slashes with underscore
    name = re.sub(r"[ /]+", "_", name)
    # keep only letters, numbers, underscore
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    # avoid leading/trailing underscores
    name = name.strip("_")
    return name or ""


def guess_ext_from_headers(resp: requests.Response, default_ext: str = ".jpg") -> str:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    # fallback from url
    path = resp.url.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return default_ext


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_rows(csv_path: Path) -> List[dict]:
    """Load CSV with Python's csv module to avoid extra deps."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def choose_label(row: dict) -> Optional[str]:
    """
    Pick a label column in priority order:
      1) scientific_name
      2) species_guess
      3) common_name (last resort; may be ambiguous)
    """
    for key in ("scientific_name", "species_guess", "common_name"):
        val = row.get(key) or ""
        val = val.strip()
        if val:
            return val
    return None


def filter_and_bucket_rows(rows: List[dict]) -> Dict[str, List[dict]]:
    """
    Keep only rows with both a label and an image_url, and bucket by class.
    """
    buckets: Dict[str, List[dict]] = defaultdict(list)
    bad_no_img = 0
    bad_no_label = 0

    for r in rows:
        img = (r.get("image_url") or "").strip()
        if not img:
            bad_no_img += 1
            continue
        label = choose_label(r)
        if not label:
            bad_no_label += 1
            continue
        class_name = sanitize_class_name(label)
        if not class_name:
            bad_no_label += 1
            continue
        # Deduplicate: some CSVs might have duplicate image_url rows
        buckets[class_name].append({"image_url": img, "row": r})

    kept_classes = len(buckets)
    total_kept = sum(len(v) for v in buckets.values())
    print(f"[info] Valid rows: {total_kept} across {kept_classes} classes "
          f"(skipped {bad_no_img} without image_url, {bad_no_label} without label).")
    return buckets


def trim_class_counts(
    buckets: Dict[str, List[dict]],
    min_per_class: int,
    max_per_class: int,
) -> Dict[str, List[dict]]:
    """
    Drop classes with < min_per_class; cap at max_per_class.
    """
    new_buckets: Dict[str, List[dict]] = {}
    dropped = 0
    for cls, items in buckets.items():
        # Use URL to dedupe within the class
        seen_urls = set()
        uniq = []
        for it in items:
            url = it["image_url"]
            if url not in seen_urls:
                uniq.append(it)
                seen_urls.add(url)

        if len(uniq) < min_per_class:
            dropped += 1
            continue
        # cap
        if len(uniq) > max_per_class:
            # shuffle for variety then slice
            random.shuffle(uniq)
            uniq = uniq[:max_per_class]
        new_buckets[cls] = uniq

    print(f"[info] Classes kept: {len(new_buckets)} (dropped {dropped} classes with < {min_per_class} samples)")
    return new_buckets


def split_train_val(items: List[dict], train_split: float) -> Tuple[List[dict], List[dict]]:
    """
    Deterministic per-class split that guarantees each class appears in BOTH splits.
    - If a class has only 1 image, we DUPLICATE that image into both train and val.
    - If a class has >=2 images, we enforce at least 1 in val and at least 1 in train.
    """
    items_sorted = sorted(items, key=lambda it: stable_hash(it["image_url"]))
    n = len(items_sorted)

    if n == 1:
        # duplicate single image into both splits (yes, this leaks; good for UI/class-count parity)
        return items_sorted[:], items_sorted[:]

    # n >= 2
    n_train = int(round(train_split * n))
    # ensure at least 1 in train and 1 in val
    n_train = max(1, min(n - 1, n_train))
    train_items = items_sorted[:n_train]
    val_items   = items_sorted[n_train:]
    if len(val_items) == 0:
        # just in case rounding pushed all into train
        val_items = [items_sorted[-1]]
        train_items = items_sorted[:-1]
    return train_items, val_items



def download_one(
    session: requests.Session,
    url: str,
    dest_path: Path,
    timeout: int = 15,
    retries: int = 3,
) -> bool:
    if dest_path.exists():
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200 or ("image" not in (r.headers.get("Content-Type") or "")):
                    last_err = RuntimeError(f"HTTP {r.status_code} {r.headers.get('Content-Type')}")
                    continue
                ext = guess_ext_from_headers(r, default_ext=dest_path.suffix or ".jpg")
                # rewrite extension if needed
                if dest_path.suffix.lower() != ext.lower():
                    dest_path = dest_path.with_suffix(ext)
                tmp = dest_path.with_suffix(dest_path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest_path)
                return True
        except Exception as e:
            last_err = e
    print(f"[warn] Failed {url} -> {dest_path} : {last_err}")
    return False


def build_targets(
    buckets: Dict[str, List[dict]],
    out_root: Path,
    train_split: float,
    workers: int,
    attribution_path: Path,
) -> Tuple[Dict[str, int], int, int]:
    """
    Create download tasks, write images to train/ and val/, and generate attribution.
    Returns species_to_idx, total_ok, total_fail counts.
    """
    species = sorted(buckets.keys())
    species_to_idx = {s: i for i, s in enumerate(species)}

    # attribution CSV header
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    with attribution_path.open("w", encoding="utf-8", newline="") as f:
        aw = csv.writer(f)
        aw.writerow(["image_url", "saved_path", "species"])

    jobs = []  # (url, dest_path, species)
    for sp, items in buckets.items():
        train_items, val_items = split_train_val(items, train_split)
        for split, split_items in (("train", train_items), ("val", val_items)):
            cls_dir = out_root / split / sp
            for it in split_items:
                url = it["image_url"]
                # file name from stable hash of url
                basename = stable_hash(url)
                dest = cls_dir / f"{basename}.jpg"
                jobs.append((url, dest, sp))

    print(f"[info] Prepared {len(jobs)} downloads. Starting (workers={workers})...")
    ok, fail = 0, 0
    lock = threading.Lock()

    # Use a single session shared across threads for connection pooling
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Plant-FYP-Downloader/1.0"})
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(download_one, session, url, dest) for (url, dest, _sp) in jobs]
            for fut, (url, dest, sp) in zip(futs, jobs):
                res = fut.result()
                if res:
                    ok += 1
                    # write attribution row
                    with lock:
                        with attribution_path.open("a", encoding="utf-8", newline="") as f:
                            aw = csv.writer(f)
                            # path relative to out_root for portability
                            rel = dest.relative_to(out_root)
                            aw.writerow([url, str(rel), sp])
                else:
                    fail += 1

    return species_to_idx, ok, fail


def save_class_maps(species_to_idx: Dict[str, int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_to_species = {int(v): k for k, v in species_to_idx.items()}
    with (out_dir / "species_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(species_to_idx, f, ensure_ascii=False, indent=2)
    with (out_dir / "idx_to_species.json").open("w", encoding="utf-8") as f:
        json.dump(idx_to_species, f, ensure_ascii=False, indent=2)
    print(f"[info] Wrote class maps to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="iNaturalist CSV -> ImageFolder downloader")
    ap.add_argument("--csv", required=True, help="Path to iNat export CSV")
    ap.add_argument("--out_dir", default="data/plants", help="Output dataset root")
    ap.add_argument("--min_per_class", type=int, default=30, help="Min images per species to keep the class")
    ap.add_argument("--max_per_class", type=int, default=400, help="Max images per species (cap)")
    ap.add_argument("--train_split", type=float, default=0.8, help="Train fraction (0..1)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent downloads")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/caps")
    args = ap.parse_args()

    random.seed(args.seed)

    csv_path = Path(args.csv)
    out_root = Path(args.out_dir)
    classmap_dir = Path("models/class_maps")
    attribution_csv = out_root / "attribution.csv"

    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        sys.exit(1)

    rows = load_rows(csv_path)
    if not rows:
        print("[error] Empty CSV or could not parse rows.")
        sys.exit(1)

    # Filter to Plantae in case CSV has more (optional; keep if your query already filtered)
    # rows = [r for r in rows if (r.get("iconic_taxon_name") or "").lower() == "plantae"]

    buckets = filter_and_bucket_rows(rows)
    buckets = trim_class_counts(buckets, args.min_per_class, args.max_per_class)

    # Download and split
    species_to_idx, ok, fail = build_targets(
        buckets=buckets,
        out_root=out_root,
        train_split=args.train_split,
        workers=args.workers,
        attribution_path=attribution_csv,
    )

    # Class maps
    save_class_maps(species_to_idx, classmap_dir)

    # Summary
    num_classes = len(species_to_idx)
    print(f"\n[done] Classes: {num_classes} | Downloaded: {ok} | Failed: {fail}")
    print(f"[paths] Dataset root: {out_root}")
    print(f"[paths] Class maps:   {classmap_dir / 'species_to_idx.json'}")
    print(f"[paths] Attribution:  {attribution_csv}")


if __name__ == "__main__":
    main()
