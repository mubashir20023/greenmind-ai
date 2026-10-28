#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a reproducible, documented dataset split for the GreenMind AI paper.

Scans one or more image folders where each image's PARENT folder is its class
(this matches ImageFolder layouts such as data/plants/train/<Species>/x.jpg and
data/plants/val/<Species>/y.jpg -- train/val folders are pooled automatically).

What it does
  1. Collects every image and its class name.
  2. Optionally removes exact duplicates (MD5 of file bytes).
  3. Optionally removes any image that also appears (by MD5) in --exclude folders,
     e.g. the images the ORIGINAL model was trained/validated on. This gives a
     test set the existing checkpoint has never seen.
  4. Keeps classes with at least --min_per_class images.
  5. Picks a per-class cap so the total is as close as possible to --total
     (balanced sampling), and samples with a fixed --seed.
  6. Splits each class into train / val / test (stratified).
  7. Writes manifest.csv (path,label,split) and split_stats.json, which are the
     exact numbers to report in the paper.
  8. Optionally (--export_dir) writes resized copies (short side 256 px, JPEG q90)
     so the dataset is small enough to zip and upload to Kaggle/Colab.

Examples (Windows, run from the repo root on G:)
  # 139k-image species dataset from everything you downloaded
  python tools/make_split.py --src G:\\data\\inat --total 139000 ^
      --out_dir G:\\data\\splits\\species139k --export_dir G:\\data\\species139k_256 --dedupe

  # Test set the EXISTING ResNet-18 never saw (exclude its train/val images)
  python tools/make_split.py --src G:\\data\\inat --exclude data\\plants ^
      --val 0 --test 1.0 --per_class_cap 20 --out_dir G:\\data\\splits\\unseen_test --dedupe
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p


def class_of(path: Path) -> str:
    return path.parent.name


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def choose_cap(counts, total):
    """Largest per-class cap c such that sum(min(n_i, c)) <= total."""
    lo, hi = 1, max(counts)
    if sum(counts) <= total:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in counts) <= total:
            lo = mid
        else:
            hi = mid - 1
    return lo


def split_counts(n, val_frac, test_frac):
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if test_frac > 0 and n_test == 0 and n >= 3:
        n_test = 1
    if val_frac > 0 and n_val == 0 and n >= 3:
        n_val = 1
    n_train = n - n_val - n_test
    if n_train < 0:  # only possible with val+test ~= 1
        n_val = max(0, n - n_test)
        n_train = 0
    return n_train, n_val, n_test


def export_resized(src: Path, dst: Path, short_side: int):
    from PIL import Image, ImageOps
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = short_side / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(dst, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="Folder(s) to scan; parent dir name = class")
    ap.add_argument("--exclude", nargs="*", default=[], help="Folder(s) whose images must NOT be used (matched by MD5)")
    ap.add_argument("--out_dir", required=True, help="Where manifest.csv and split_stats.json go")
    ap.add_argument("--total", type=int, default=0, help="Target total images (0 = use everything after caps)")
    ap.add_argument("--per_class_cap", type=int, default=0, help="Fixed per-class cap (overrides --total balancing)")
    ap.add_argument("--min_per_class", type=int, default=30)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicate files (MD5)")
    ap.add_argument("--export_dir", default="", help="Optional: write resized copies here (ImageFolder by split)")
    ap.add_argument("--short_side", type=int, default=256)
    args = ap.parse_args()

    if args.val + args.test > 1.0 + 1e-9:
        sys.exit("--val + --test must be <= 1")

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) excluded hashes
    excluded = set()
    for ex in args.exclude:
        exp = Path(ex)
        print(f"[exclude] hashing {exp} ...")
        for i, p in enumerate(iter_images(exp), 1):
            excluded.add(md5_of(p))
            if i % 20000 == 0:
                print(f"  {i} files hashed")
    if args.exclude:
        print(f"[exclude] {len(excluded)} unique images will be skipped")

    # 2) collect
    by_class = defaultdict(list)
    seen = set()
    n_scanned = n_dup = n_excl = 0
    need_hash = args.dedupe or bool(excluded)
    for s in args.src:
        sp = Path(s)
        print(f"[scan] {sp}")
        for p in iter_images(sp):
            n_scanned += 1
            if need_hash:
                h = md5_of(p)
                if h in excluded:
                    n_excl += 1
                    continue
                if args.dedupe:
                    if h in seen:
                        n_dup += 1
                        continue
                    seen.add(h)
            cls = class_of(p)
            if cls.lower() in SPLIT_DIR_NAMES:  # image sitting directly in train/ etc.
                continue
            by_class[cls].append(p)
            if n_scanned % 50000 == 0:
                print(f"  {n_scanned} scanned")
    print(f"[scan] {n_scanned} images, {len(by_class)} classes, dup={n_dup}, excluded={n_excl}")

    # 3) class filter
    kept = {c: sorted(v) for c, v in by_class.items() if len(v) >= args.min_per_class}
    dropped_classes = len(by_class) - len(kept)
    if not kept:
        sys.exit("No class has enough images; lower --min_per_class")

    # 4) cap
    counts = [len(v) for v in kept.values()]
    if args.per_class_cap > 0:
        cap = args.per_class_cap
    elif args.total > 0:
        cap = choose_cap(counts, args.total)
    else:
        cap = max(counts)
    print(f"[cap] per-class cap = {cap}")

    # 5) sample + split
    rows = []
    split_total = Counter()
    for cls in sorted(kept):
        files = kept[cls][:]
        rng.shuffle(files)
        files = files[:cap]
        n_tr, n_va, n_te = split_counts(len(files), args.val, args.test)
        parts = (["train"] * n_tr) + (["val"] * n_va) + (["test"] * n_te)
        for p, sp in zip(files, parts):
            rows.append((str(p.resolve()), cls, sp))
            split_total[sp] += 1

    # 6) write manifest
    man = out_dir / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)

    per_class = Counter(r[1] for r in rows)
    pc = sorted(per_class.values())
    stats = {
        "sources": args.src,
        "excluded_sources": args.exclude,
        "seed": args.seed,
        "scanned_images": n_scanned,
        "exact_duplicates_removed": n_dup,
        "excluded_overlap_removed": n_excl,
        "classes_found": len(by_class),
        "classes_dropped_min_per_class": dropped_classes,
        "min_per_class": args.min_per_class,
        "classes_kept": len(kept),
        "per_class_cap": cap,
        "total_images": len(rows),
        "split_sizes": dict(split_total),
        "split_fractions": {"val": args.val, "test": args.test},
        "images_per_class": {"min": pc[0], "median": pc[len(pc) // 2], "max": pc[-1]},
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(sorted(kept), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"[done] {man}")

    # 7) optional export
    if args.export_dir:
        ex = Path(args.export_dir)
        print(f"[export] writing resized copies to {ex}")
        new_rows = []
        for i, (p, cls, sp) in enumerate(rows, 1):
            src = Path(p)
            dst = ex / sp / cls / (src.stem + ".jpg")
            k = 1
            while dst.exists():
                dst = ex / sp / cls / f"{src.stem}_{k}.jpg"
                k += 1
            try:
                export_resized(src, dst, args.short_side)
            except Exception as e:  # corrupt file: keep the copy honest by skipping it
                print(f"  [skip] {src}: {e}")
                continue
            new_rows.append((dst.relative_to(ex).as_posix(), cls, sp))
            if i % 10000 == 0:
                print(f"  {i}/{len(rows)}")
        with (ex / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "split"])
            w.writerows(new_rows)
        shutil.copy(out_dir / "split_stats.json", ex / "split_stats.json")
        shutil.copy(out_dir / "classes.json", ex / "classes.json")
        if len(new_rows) != len(rows):
            print(f"[export] {len(rows) - len(new_rows)} unreadable images skipped; "
                  f"use {ex / 'manifest.csv'} as the final manifest")
        print(f"[export] done. Zip {ex} and upload it as a Kaggle dataset.")


if __name__ == "__main__":
    main()
