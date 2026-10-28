#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-create the GreenMind species dataset from iNaturalist, with a TEMPORAL hold-out.

For every species in a class map (default: models/class_maps/idx_to_species.json,
the 1,106 classes of the deployed ResNet-18) it downloads research-grade,
Creative-Commons-licensed observations into two pools:

  pool_a  observations CREATED BEFORE --cutoff   -> train / val / in-distribution test
  pool_b  observations CREATED ON/AFTER --after  -> temporal hold-out test

The deployed checkpoint was committed on 2025-09-02, so with --after 2025-09-03
pool_b cannot contain any image that model was trained on. That makes it a clean
test set for the existing model as well as for retrained ones.

One photo per observation is used (the first one), so the same plant photographed
several times never ends up in both train and test.

Output layout (resized to --short_side px, JPEG q90):
  OUT/pool_a/<Species_name>/<obs_id>.jpg
  OUT/pool_b/<Species_name>/<obs_id>.jpg
  OUT/records.csv          one row per downloaded image (license + attribution)
  OUT/taxa.csv             species name -> iNat taxon id (or why it was skipped)
  OUT/fetch_stats.json

Usage (Kaggle / Colab with Internet ON):
  python tools/inat_fetch_species.py --out /kaggle/working/inat --per_class_a 150 --per_class_b 20

Respects iNaturalist API etiquette: ~1 request/second and a download budget in GB/hour.
Resumable: re-running skips species already finished (taxa.csv/records.csv are appended).
"""

import argparse
import csv
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

API = "https://api.inaturalist.org/v1"
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa"
UA = "GreenMind-AI-research/1.0 (github.com/mubashir20023/greenmind-ai)"


class RateLimiter:
    def __init__(self, per_sec):
        self.dt = 1.0 / per_sec
        self.t = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.dt


class ByteBudget:
    """Keeps the average download rate under gb_per_hour."""

    def __init__(self, gb_per_hour):
        self.bps = gb_per_hour * 1e9 / 3600
        self.start = time.time()
        self.bytes = 0
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.bytes += n
            ahead = self.bytes / self.bps - (time.time() - self.start)
        if ahead > 0:
            time.sleep(ahead)


def api_get(sess, rl, path, params, tries=5):
    for k in range(tries):
        rl.wait()
        try:
            r = sess.get(f"{API}/{path}", params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
            print(f"  [retry] {path}: {e}")
    return None


def resolve_taxon(sess, rl, name):
    sci = name.replace("_", " ")
    js = api_get(sess, rl, "taxa", {"q": sci, "rank": "species,hybrid,variety,subspecies", "per_page": 30})
    for t in js.get("results", []):
        if t.get("name", "").lower() == sci.lower():
            return t["id"], t.get("is_active", True), "exact"
    # synonym: search may return the currently accepted taxon with the name as a match
    for t in js.get("results", []):
        if (t.get("matched_term") or "").lower() == sci.lower():
            return t["id"], t.get("is_active", True), f"synonym->{t.get('name')}"
    return None, None, "not_found"


def fetch_obs(sess, rl, taxon_id, n, created_d1=None, created_d2=None):
    """Most recent research-grade CC observations in the window, one photo each."""
    out, id_below = [], None
    while len(out) < n:
        params = {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
                  "photo_license": LICENSES, "per_page": 200, "order_by": "id", "order": "desc"}
        if created_d1:
            params["created_d1"] = created_d1
        if created_d2:
            params["created_d2"] = created_d2
        if id_below:
            params["id_below"] = id_below
        js = api_get(sess, rl, "observations", params)
        res = js.get("results", [])
        for o in res:
            ph = next((p for p in o.get("photos") or []
                       if (p.get("license_code") or "").lower() in LICENSES.split(",") and p.get("url")), None)
            if not ph:
                continue
            out.append({
                "obs_id": o["id"], "photo_id": ph["id"],
                "url": ph["url"].replace("/square.", "/medium.").replace("/thumb.", "/medium."),
                "license": ph.get("license_code"), "attribution": ph.get("attribution", ""),
                "created_at": o.get("created_at", ""), "observed_on": o.get("observed_on") or "",
                "user": (o.get("user") or {}).get("login", ""),
                "taxon_name": (o.get("taxon") or {}).get("name", ""),
            })
            if len(out) >= n:
                break
        if len(res) < 200:
            break
        id_below = res[-1]["id"]
    return out


def download(sess, budget, url, dst, short_side):
    for k in range(3):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            budget.add(len(r.content))
            with Image.open(io.BytesIO(r.content)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                s = short_side / min(w, h)
                if s < 1:
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, "JPEG", quality=90)
            return True
        except Exception:
            time.sleep(2 * (k + 1))
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="models/class_maps/idx_to_species.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", default="2025-08-31", help="pool_a: created on/before this date")
    ap.add_argument("--after", default="2025-09-03", help="pool_b: created on/after this date")
    ap.add_argument("--per_class_a", type=int, default=150)
    ap.add_argument("--per_class_b", type=int, default=20)
    ap.add_argument("--short_side", type=int, default=256)
    ap.add_argument("--api_rps", type=float, default=1.0)
    ap.add_argument("--gb_per_hour", type=float, default=4.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_species", type=int, default=0, help="Debug: only first N species")
    args = ap.parse_args()

    cm = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    species = [cm[str(i)] for i in range(len(cm))] if isinstance(cm, dict) else list(cm)
    if args.limit_species:
        species = species[: args.limit_species]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    sess.mount("https://", adapter)
    rl = RateLimiter(args.api_rps)
    budget = ByteBudget(args.gb_per_hour)

    taxa_csv, rec_csv = out / "taxa.csv", out / "records.csv"
    new_files = not taxa_csv.exists()
    done = set()
    if taxa_csv.exists():
        with taxa_csv.open(encoding="utf-8") as f:
            done = {r["label"] for r in csv.DictReader(f)}
    tf = taxa_csv.open("a", newline="", encoding="utf-8")
    rf = rec_csv.open("a", newline="", encoding="utf-8")
    tw, rw = csv.writer(tf), csv.writer(rf)
    if new_files:
        tw.writerow(["label", "taxon_id", "match", "n_pool_a", "n_pool_b"])
        rw.writerow(["pool", "label", "file", "obs_id", "photo_id", "taxon_name", "license", "attribution",
                     "user", "created_at", "observed_on", "url"])

    pool = ThreadPoolExecutor(args.workers)
    t0 = time.time()
    for i, label in enumerate(species, 1):
        if label in done:
            continue
        try:
            tid, active, match = resolve_taxon(sess, rl, label)
        except Exception as e:
            tid, match = None, f"error:{e}"
        if not tid:
            tw.writerow([label, "", match, 0, 0])
            tf.flush()
            print(f"[{i}/{len(species)}] {label}: {match}")
            continue
        counts = {}
        for pname, n, d1, d2 in (("pool_a", args.per_class_a, None, args.cutoff),
                                 ("pool_b", args.per_class_b, args.after, None)):
            obs = fetch_obs(sess, rl, tid, n, d1, d2)
            jobs = []
            for o in obs:
                dst = out / pname / label / f"{o['obs_id']}.jpg"
                if dst.exists():
                    jobs.append((o, dst, None))
                else:
                    jobs.append((o, dst, pool.submit(download, sess, budget, o["url"], dst, args.short_side)))
            ok = 0
            for o, dst, fut in jobs:
                if fut is None or fut.result():
                    ok += 1
                    rw.writerow([pname, label, dst.relative_to(out).as_posix(), o["obs_id"], o["photo_id"],
                                 o["taxon_name"], o["license"], o["attribution"], o["user"],
                                 o["created_at"], o["observed_on"], o["url"]])
            counts[pname] = ok
        tw.writerow([label, tid, match, counts["pool_a"], counts["pool_b"]])
        tf.flush()
        rf.flush()
        el = (time.time() - t0) / 60
        print(f"[{i}/{len(species)}] {label}: A={counts['pool_a']} B={counts['pool_b']} ({match}) {el:.0f} min",
              flush=True)

    pool.shutdown()
    tf.close()
    rf.close()

    with taxa_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stats = {
        "species_requested": len(species),
        "species_resolved": sum(1 for r in rows if r["taxon_id"]),
        "species_not_found": [r["label"] for r in rows if not r["taxon_id"]],
        "images_pool_a": sum(int(r["n_pool_a"]) for r in rows),
        "images_pool_b": sum(int(r["n_pool_b"]) for r in rows),
        "cutoff_pool_a_created_on_or_before": args.cutoff,
        "pool_b_created_on_or_after": args.after,
        "licenses": LICENSES, "quality_grade": "research", "one_photo_per_observation": True,
        "short_side_px": args.short_side,
        "fetched_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
    }
    (out / "fetch_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "species_not_found"}, indent=2))
    print(f"not found: {len(stats['species_not_found'])}")


if __name__ == "__main__":
    main()
