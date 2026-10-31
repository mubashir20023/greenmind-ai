#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extra disease experiments for the GreenMind paper (used by notebooks/04_disease_improvements.ipynb).

Subcommands
  field-manifest   PlantVillage manifest + PlantDoc-train field images (mapped to PlantVillage
                   classes, de-duplicated against PlantDoc-test by MD5) -> mixed training manifest,
                   plus a field calibration manifest.
  yolo-crops       Crop the most confident 'leaf' detection of every image with the GreenMind YOLO
                   detector (fallback: full image) -> ImageFolder for evaluate.py.
  calibrate        Temperature scaling fitted on a calibration set; ECE/NLL before and after on a
                   test set; selective prediction (risk-coverage) table and curve.
  occlude          Evaluate on images whose central region is blacked out (shortcut / background test).
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def images_in(d):
    return sorted(p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def readable(p):
    from PIL import Image
    try:
        with Image.open(p) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- field-manifest
def cmd_field_manifest(a):
    lm = json.loads(Path(a.label_map).read_text(encoding="utf-8"))
    test_hashes = set()
    for folder in lm:
        d = Path(a.pd_test) / folder
        if d.is_dir():
            test_hashes.update(md5(p) for p in images_in(d))

    rng = random.Random(a.seed)
    field = defaultdict(list)
    dropped_dup = dropped_bad = 0
    seen = set()
    for folder, label in lm.items():
        d = Path(a.pd_train) / folder
        if not d.is_dir():
            continue
        for p in images_in(d):
            h = md5(p)
            if h in test_hashes or h in seen:
                dropped_dup += 1
                continue
            if not readable(p):
                dropped_bad += 1
                continue
            seen.add(h)
            field[label].append(str(p.resolve()))

    with open(a.pv_manifest, newline="", encoding="utf-8") as f:
        pv_rows = [(r["path"], r["label"], r["split"]) for r in csv.DictReader(f)]
    pv_base = Path(a.pv_manifest).parent
    pv_rows = [(p if Path(p).is_absolute() else str((pv_base / p).resolve()), l, s) for p, l, s in pv_rows]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mixed, calib = list(pv_rows), []
    n_tr = n_va = 0
    for label, paths in sorted(field.items()):
        rng.shuffle(paths)
        k = max(1, round(len(paths) * a.val_frac)) if len(paths) >= 5 else 0
        for p in paths[:k]:
            mixed.append((p, label, "val"))
            n_va += 1
        for p in paths[k:]:
            for _ in range(a.oversample):
                mixed.append((p, label, "train"))
            n_tr += 1
        for p in paths:
            calib.append((p, label, "calib"))

    for name, rows in (("mixed_manifest.csv", mixed), ("field_calib_manifest.csv", calib)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "split"])
            w.writerows(rows)
    stats = {
        "plantdoc_train_images_used": sum(len(v) for v in field.values()),
        "plantdoc_train_classes": len(field),
        "per_class": {k: len(v) for k, v in sorted(field.items())},
        "field_train_unique": n_tr, "field_val": n_va, "oversample": a.oversample,
        "dropped_duplicate_of_test_or_repeat": dropped_dup, "dropped_unreadable": dropped_bad,
        "mixed_split_sizes": dict(Counter(s for _, _, s in mixed)),
    }
    (out / "field_manifest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- yolo-crops
def cmd_yolo_crops(a):
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    lm = json.loads(Path(a.label_map).read_text(encoding="utf-8")) if a.label_map else None
    model = YOLO(a.weights)
    names = {int(k): str(v).lower() for k, v in dict(model.names).items()}
    leaf_ids = {k for k, v in names.items() if v == a.organ}
    print("detector classes:", names, "| using:", a.organ, leaf_ids)
    out = Path(a.out)
    n = n_det = 0
    for cls_dir in sorted(d for d in Path(a.src).iterdir() if d.is_dir()):
        if lm is not None and cls_dir.name not in lm:
            continue
        for p in images_in(cls_dir):
            try:
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            except Exception:
                continue
            n += 1
            W, H = im.size
            crop = im
            r = model.predict(im, conf=a.conf, verbose=False)[0]
            best = None
            for b in r.boxes:
                c = int(b.cls.item())
                if c in leaf_ids and (best is None or float(b.conf) > best[0]):
                    best = (float(b.conf), [float(v) for v in b.xyxy[0].tolist()])
            if best is not None:
                x1, y1, x2, y2 = best[1]
                pw, ph = (x2 - x1) * a.pad, (y2 - y1) * a.pad
                x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
                x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
                if (x2 - x1) * (y2 - y1) >= a.min_area * W * H:
                    crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
                    n_det += 1
            dst = out / cls_dir.name / (p.stem + ".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, "JPEG", quality=95)
    stats = {"images": n, "with_leaf_crop": n_det, "fallback_full_image": n - n_det,
             "organ": a.organ, "conf": a.conf, "pad": a.pad}
    (out / "crop_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- shared model helpers
def _load(a, classes):
    import torch
    from evaluate import load_model
    from gm_data import pick_device
    dev = pick_device("auto")
    return load_model(a.ckpt, a.arch, len(classes), dev), dev


def _rows(a, manifest=None, split=None, folder=None):
    from gm_data import read_imagefolder, read_manifest
    if folder:
        rows = read_imagefolder(folder)
        if a.label_map:
            lm = json.loads(Path(a.label_map).read_text(encoding="utf-8"))
            rows = [(p, lm.get(l, l)) for p, l in rows]
        return rows
    return read_manifest(manifest, split)


def _logits(model, dev, rows, classes, a, pil_hook=None):
    import torch
    from torch.utils.data import DataLoader
    from gm_data import ManifestDataset, build_transforms
    t = build_transforms(a.img_size, a.norm, False, a.resize_mode)
    if pil_hook is not None:
        base = t
        t = lambda im: base(pil_hook(im))  # noqa: E731
    ds = ManifestDataset(rows, classes, t)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers)
    L, Y = [], []
    with torch.inference_mode():
        for x, y in ld:
            L.append(model(x.to(dev)).float().cpu())
            Y.append(y)
    return torch.cat(L), torch.cat(Y), ds.n_unknown


def _ece(conf, correct, bins=15):
    import numpy as np
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _summ(logits, y, T=1.0):
    import torch
    import torch.nn.functional as F
    p = torch.softmax(logits / T, 1)
    conf, pred = p.max(1)
    correct = (pred == y).float()
    return {
        "acc": float(correct.mean()), "nll": float(F.cross_entropy(logits / T, y)),
        "ece": _ece(conf.numpy(), correct.numpy()), "mean_conf": float(conf.mean()),
    }, conf.numpy(), correct.numpy()


# --------------------------------------------------------------------------- calibrate
def cmd_calibrate(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gm_data import load_classes
    classes = load_classes(a.classes)
    model, dev = _load(a, classes)
    cal_rows = _rows(a, a.calib_manifest, a.calib_split, a.calib_imagefolder)
    test_rows = _rows(a, folder=a.test_imagefolder)
    Lc, Yc, uc = _logits(model, dev, cal_rows, classes, a)
    Lt, Yt, ut = _logits(model, dev, test_rows, classes, a)

    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Lc / logT.exp(), Yc)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(logT.exp())

    before, _, _ = _summ(Lt, Yt, 1.0)
    after, conf, correct = _summ(Lt, Yt, T)
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    cov = np.arange(1, len(order) + 1) / len(order)
    table = {}
    for c in (1.0, 0.9, 0.75, 0.5, 0.25):
        k = max(1, int(round(c * len(order))))
        table[f"coverage_{int(c*100)}"] = float(cum_acc[k - 1])
    thr = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = conf >= t
        thr[str(t)] = {"coverage": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None}
    res = {
        "calibration_set": a.calib_imagefolder or f"{a.calib_manifest}[{a.calib_split}]",
        "n_calib": int(len(Yc)), "n_test": int(len(Yt)), "calib_unknown_labels": uc, "test_unknown_labels": ut,
        "temperature": T, "test_before": before, "test_after": after,
        "accuracy_at_coverage": table, "confidence_threshold": thr,
        "aurc": float(np.mean(1 - cum_acc)),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (out / "risk_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage", "accuracy"])
        for c_, a_ in zip(cov, cum_acc):
            w.writerow([f"{c_:.4f}", f"{a_:.4f}"])
    print(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- occlude
def cmd_occlude(a):
    from PIL import ImageDraw
    from gm_data import load_classes
    classes = load_classes(a.classes)
    model, dev = _load(a, classes)
    rows = _rows(a, a.manifest, a.split)
    res = {"n": None, "chance": 1.0 / len(classes)}
    counts = Counter(l for _, l in rows)
    res["majority_class_rate"] = max(counts.values()) / sum(counts.values())
    for frac in [0.0] + [float(x) for x in a.fracs.split(",")]:
        def hook(im, frac=frac):
            if frac <= 0:
                return im
            im = im.copy()
            W, H = im.size
            w, h = W * frac, H * frac
            ImageDraw.Draw(im).rectangle([(W - w) / 2, (H - h) / 2, (W + w) / 2, (H + h) / 2], fill=(0, 0, 0))
            return im
        L, Y, _ = _logits(model, dev, rows, classes, a, pil_hook=hook)
        s, _, _ = _summ(L, Y)
        res["n"] = int(len(Y))
        res[f"center_occluded_{frac:.2f}"] = {"acc": s["acc"], "mean_conf": s["mean_conf"]}
        print(frac, s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "occlusion.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("field-manifest")
    f.add_argument("--pv_manifest", required=True)
    f.add_argument("--pd_train", required=True)
    f.add_argument("--pd_test", required=True)
    f.add_argument("--label_map", required=True)
    f.add_argument("--oversample", type=int, default=5)
    f.add_argument("--val_frac", type=float, default=0.15)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--out", required=True)

    y = sub.add_parser("yolo-crops")
    y.add_argument("--weights", required=True)
    y.add_argument("--src", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("--label_map", default="")
    y.add_argument("--organ", default="leaf")
    y.add_argument("--conf", type=float, default=0.25)
    y.add_argument("--pad", type=float, default=0.10)
    y.add_argument("--min_area", type=float, default=0.04)

    for name in ("calibrate", "occlude"):
        c = sub.add_parser(name)
        c.add_argument("--ckpt", required=True)
        c.add_argument("--arch", default="efficientnet_b0")
        c.add_argument("--classes", required=True)
        c.add_argument("--norm", default="half")
        c.add_argument("--resize_mode", default="square")
        c.add_argument("--img_size", type=int, default=224)
        c.add_argument("--workers", type=int, default=4)
        c.add_argument("--label_map", default="")
        c.add_argument("--out", required=True)
        if name == "calibrate":
            c.add_argument("--calib_manifest", default="")
            c.add_argument("--calib_split", default="calib")
            c.add_argument("--calib_imagefolder", default="")
            c.add_argument("--test_imagefolder", required=True)
        else:
            c.add_argument("--manifest", required=True)
            c.add_argument("--split", default="test")
            c.add_argument("--fracs", default="0.5,0.7")

    a = ap.parse_args()
    {"field-manifest": cmd_field_manifest, "yolo-crops": cmd_yolo_crops,
     "calibrate": cmd_calibrate, "occlude": cmd_occlude}[a.cmd](a)


if __name__ == "__main__":
    main()
