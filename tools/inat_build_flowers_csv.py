#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build/Resume a CSV of (image_url, scientific_name, common_name) for top-K flowering plant species from iNaturalist.

Features:
- RESUMABLE: If --resume and the CSV exists, it loads current rows, dedupes URLs,
  and continues fetching until each species has up to --max_per_class rows.
- Skips species with < --min_per_class usable photos at the end of their crawl.
- Optional species-list cache to avoid refetching the top species on resume.

Usage (fresh):
  python tools/inat_build_flowers_csv.py --out_csv raw_data/flowers_1000.csv --num_species 1000 \
      --min_per_class 80 --max_per_class 400 --licenses CC0,CC-BY,CC-BY-SA,CC-BY-NC --sleep 0.4

Usage (resume):
  python tools/inat_build_flowers_csv.py --out_csv raw_data/flowers_1000.csv --num_species 1000 \
      --min_per_class 80 --max_per_class 400 --resume --species_cache raw_data/flowers_species_cache.json
"""

import csv
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# iNat API
SPECIES_COUNTS_URL = "https://api.inaturalist.org/v1/observations/species_counts"
OBSERVATIONS_URL   = "https://api.inaturalist.org/v1/observations"

def rq_session(retries: int = 3, timeout: int = 30) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "flowers-builder/1.0"})
    s.request_timeout = timeout
    s.retries = retries
    return s

def safe_get(sess: requests.Session, url: str, params: dict, timeout: int = 30):
    last = None
    for _ in range(getattr(sess, "retries", 3)):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last

def fetch_top_species(sess: requests.Session, num_species: int, root_taxon_id: int, sleep: float) -> List[Dict]:
    out = []
    page = 1
    per_page = 200
    while len(out) < num_species:
        params = dict(
            quality_grade="research",
            rank="species",
            taxon_id=root_taxon_id,
            photos="true",
            per_page=per_page,
            page=page,
            order_by="observations_count",
            order="desc",
        )
        r = safe_get(sess, SPECIES_COUNTS_URL, params)
        js = r.json()
        results = js.get("results", [])
        if not results:
            break
        out.extend(results)
        page += 1
        time.sleep(sleep)
    return out[:num_species]

def fetch_photos_for_taxon(
    sess: requests.Session,
    taxon_id: int,
    licenses: Set[str],
    sleep: float,
    start_page: int = 1,
    max_pages: int = 999,
) -> Tuple[List[Dict], int]:
    """
    Returns (rows, last_page_seen). Each row: {image_url, scientific_name, common_name}.
    We stream pages newestâ†’older and stop when pages end or max_pages reached.
    """
    rows = []
    page = max(1, start_page)
    per_page = 200
    while page < start_page + max_pages:
        params = dict(
            taxon_id=taxon_id,
            quality_grade="research",
            photos="true",
            per_page=per_page,
            page=page,
            order="desc",
            order_by="created_at",
            locale="en",
        )
        r = safe_get(sess, OBSERVATIONS_URL, params)
        js = r.json()
        results = js.get("results", [])
        if not results:
            return rows, page
        for obs in results:
            taxon = obs.get("taxon") or {}
            sci = taxon.get("name") or ""
            com = taxon.get("preferred_common_name") or ""
            for ph in (obs.get("photos") or []):
                lic = (ph.get("license_code") or "").upper().replace("CC-", "CC-")
                if lic and licenses and lic not in licenses:
                    continue
                url = ph.get("url") or ""
                if not url:
                    continue
                # iNat thumbnails â†’ larger image
                img_url = url.replace("square.", "large.").replace("small.", "large.")
                rows.append(dict(image_url=img_url, scientific_name=sci, common_name=com))
        page += 1
        time.sleep(sleep)
        if len(results) < per_page:
            return rows, page
    return rows, page

def load_existing(out_csv: Path) -> Tuple[Set[str], Dict[str, int]]:
    """
    Returns (seen_urls, per_species_counts) from an existing CSV; empty if none.
    """
    seen = set()
    per_species = defaultdict(int)
    if not out_csv.exists():
        return seen, per_species
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            url = (row.get("image_url") or "").strip()
            sci = (row.get("scientific_name") or "").strip()
            if url:
                seen.add(url)
            if sci:
                per_species[sci] += 1
    return seen, per_species

def write_rows_atomic(out_csv: Path, header: List[str], new_rows: List[List[str]]):
    """
    Append rows atomically (tmp â†’ rename). If file doesn't exist, write header first.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_csv.exists() else "w"
    tmp = out_csv.with_suffix(out_csv.suffix + ".part")
    # Weâ€™ll append directly; on Windows atomic replace is limited, but we still try to avoid partial headers.
    if mode == "w":
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in new_rows:
                w.writerow(r)
        tmp.replace(out_csv)
    else:
        # Append safely (if interrupted, a partial .part won't affect the main CSV)
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for r in new_rows:
                w.writerow(r)
        # Append tmp â†’ out, then delete tmp
        with out_csv.open("a", encoding="utf-8", newline="") as f_out, tmp.open("r", encoding="utf-8", newline="") as f_in:
            f_out.write(f_in.read())
        tmp.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--num_species", type=int, default=1000)
    ap.add_argument("--min_per_class", type=int, default=80)
    ap.add_argument("--max_per_class", type=int, default=400)
    ap.add_argument("--root_taxon_id", type=int, default=47126, help="Plantae=47126; pass Angiosperms clade ID if known")
    ap.add_argument("--licenses", default="CC0,CC-BY,CC-BY-SA,CC-BY-NC",
                    help="Comma-separated iNat photo license codes to allow")
    ap.add_argument("--sleep", type=float, default=0.4, help="Delay between requests")
    ap.add_argument("--resume", action="store_true", help="Resume into existing CSV if present")
    ap.add_argument("--species_cache", default="", help="Optional JSON file to cache the top-species list")
    ap.add_argument("--pages_per_species_step", type=int, default=2,
                    help="When resuming/filling, fetch this many pages per pass to spread load")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    licenses = {s.strip().upper() for s in args.licenses.split(",") if s.strip()}

    sess = rq_session()

    # Load or fetch the species list (cacheable)
    species_list_path = Path(args.species_cache) if args.species_cache else None
    if species_list_path and species_list_path.exists():
        species_rows = json.loads(species_list_path.read_text(encoding="utf-8"))
        print(f"[info] Loaded {len(species_rows)} species from cache: {species_list_path}")
    else:
        print(f"[info] Fetching top {args.num_species} species (taxon_id={args.root_taxon_id})...")
        species_rows = fetch_top_species(sess, args.num_species, args.root_taxon_id, args.sleep)
        print(f"[info] Got {len(species_rows)} species candidates")
        if species_list_path:
            species_list_path.parent.mkdir(parents=True, exist_ok=True)
            species_list_path.write_text(json.dumps(species_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[cache] Wrote species cache â†’ {species_list_path}")

    # Resume: load existing URLs and per-species counts
    seen_urls, per_species_counts = load_existing(out_csv) if args.resume else (set(), defaultdict(int))

    header = ["image_url", "scientific_name", "common_name"]
    kept_species = 0
    new_rows_buf: List[List[str]] = []

    # Pass 1: make at least min_per_class for each species (or skip if impossible)
    for i, sp in enumerate(species_rows, start=1):
        tax = sp.get("taxon") or {}
        tid = tax.get("id")
        sci = (tax.get("name") or "").strip()
        com = (tax.get("preferred_common_name") or "").strip()
        if not tid or not sci:
            continue

        have = per_species_counts.get(sci, 0)
        need_min = max(0, args.min_per_class - have)
        need_max = max(0, args.max_per_class - have)
        if need_max <= 0:
            # already full
            kept_species += 1
            print(f"[full] {i:4d}: {sci} (taxon_id={tid}) have={have} (â‰¥ max {args.max_per_class})")
            continue

        # Fetch in small steps to be resume-friendly and polite
        total_added = 0
        page = 1
        while total_added < need_max:
            rows, page = fetch_photos_for_taxon(sess, tid, licenses, args.sleep, start_page=page, max_pages=args.pages_per_species_step)
            if not rows:
                break
            # Filter + dedupe
            added_this_round = 0
            for r in rows:
                url = r["image_url"]
                if url in seen_urls:
                    continue
                new_rows_buf.append([url, r["scientific_name"], r["common_name"]])
                seen_urls.add(url)
                per_species_counts[sci] += 1
                total_added += 1
                added_this_round += 1
                if total_added >= need_max:
                    break

            # Flush to disk in chunks to survive interruptions
            if added_this_round > 0 and len(new_rows_buf) >= 500:
                write_rows_atomic(out_csv, header, new_rows_buf)
                new_rows_buf.clear()

            # If we didnâ€™t get any new additions this pass, likely exhausted
            if added_this_round == 0:
                break

        have_after = per_species_counts.get(sci, 0)
        if have_after >= args.min_per_class:
            kept_species += 1
            print(f"[keep] {i:4d}: {sci} (taxon_id={tid}) rows={have_after} (added {have_after - have})")
        else:
            print(f"[skip] {i:4d}: {sci} (taxon_id={tid}) < {args.min_per_class} usable photos (have {have_after})")

        # Periodic flush
        if new_rows_buf:
            write_rows_atomic(out_csv, header, new_rows_buf)
            new_rows_buf.clear()

    # Final flush (just in case)
    if new_rows_buf:
        write_rows_atomic(out_csv, header, new_rows_buf)
        new_rows_buf.clear()

    print(f"[done] Species kept: {kept_species} | CSV: {out_csv}")

if __name__ == "__main__":
    main()
