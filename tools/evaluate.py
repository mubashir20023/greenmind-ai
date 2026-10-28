#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a GreenMind classifier and write every number/figure the paper needs.

Data source (pick one):
  --manifest M --split test          rows from a manifest.csv
  --imagefolder DIR                  DIR/<class_name>/*.jpg (e.g. your own Karachi photos)
      --label_map map.json           optional {"folder_name": "model_class_name"}

Model:
  --ckpt best.pth --arch resnet18    state_dict (species model, or train_classifier output)
  --ckpt model_full.pt               whole saved model (old app/health.py format); --arch not needed
  --classes classes.json             list, or idx->name dict (models/class_maps/idx_to_species.json)

Outputs in --out:
  metrics.json        top-1/top-5, macro & weighted P/R/F1, balanced acc, ECE,
                      95% bootstrap CI for top-1, CPU latency (ms/image)
  per_class.csv       precision/recall/F1/support per class
  top_confusions.csv  most frequent (true -> predicted) mistakes
  predictions.csv     one row per image
  confusion_matrix.png, reliability.png

Examples
  # existing species model on an unseen test set
  python tools/evaluate.py --ckpt models/checkpoints/plant_id_resnet18_best.pth --arch resnet18 ^
      --classes models/class_maps/idx_to_species.json --manifest G:/data/splits/unseen_test/manifest.csv ^
      --split test --out results/species_existing_unseen

  # disease model on real-world photos
  python tools/evaluate.py --ckpt runs/disease_effb0/best.pth --arch efficientnet_b0 --norm half ^
      --resize_mode square --classes runs/disease_effb0/classes.json ^
      --imagefolder G:/data/karachi_leaves --out results/disease_karachi
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm_data import (ManifestDataset, build_transforms, load_classes, pick_device,  # noqa: E402
                     read_imagefolder, read_manifest)


def load_model(ckpt, arch, num_classes, device):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        import timm
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
        if not arch:
            sys.exit("--arch is required for a state_dict checkpoint")
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if missing or unexpected:
            print(f"[warn] missing={len(missing)} unexpected={len(unexpected)} keys "
                  f"(check --arch and --classes match the checkpoint)")
    return model.eval().to(device)


def ece_score(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - c)
            rows.append((lo, hi, int(m.sum()), float(acc), float(c)))
    return float(ece), rows


@torch.no_grad()
def latency_ms(model, img_size, n=50):
    m = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        m(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None)
    ap.add_argument("--imagefolder")
    ap.add_argument("--label_map", help="JSON {folder_name: class_name}")
    ap.add_argument("--norm", choices=["imagenet", "half"], default="imagenet")
    ap.add_argument("--resize_mode", choices=["crop", "square"], default="crop")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--no_latency", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    classes = load_classes(args.classes)

    if args.imagefolder:
        rows = read_imagefolder(args.imagefolder)
        if args.label_map:
            lm = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
            rows = [(p, lm.get(l, l)) for p, l in rows]
        source = f"imagefolder:{args.imagefolder}"
    elif args.manifest:
        rows = read_manifest(args.manifest, args.split, args.root)
        source = f"manifest:{args.manifest}[{args.split}]"
    else:
        sys.exit("Give --manifest or --imagefolder")

    ds = ManifestDataset(rows, classes, build_transforms(args.img_size, args.norm, False, args.resize_mode))
    if ds.n_unknown:
        unk = Counter(l for _, l in rows if l not in set(classes))
        print(f"[warn] {ds.n_unknown} images have labels the model does not know; excluded. "
              f"Examples: {list(unk.items())[:5]}")
    if len(ds) == 0:
        sys.exit("No evaluable images.")
    print(f"[data] {len(ds)} images, {len(set(ds.targets))} classes present, model has {len(classes)}")

    model = load_model(args.ckpt, args.arch, len(classes), device)
    ld = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)

    probs_all, ys = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(ld):
        p = torch.softmax(model(x.to(device)).float(), 1).cpu()
        probs_all.append(p)
        ys.append(y)
        if i % 20 == 0:
            print(f"  batch {i}/{len(ld)}", flush=True)
    probs = torch.cat(probs_all).numpy()
    y = torch.cat(ys).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(float)
    k = min(5, probs.shape[1])
    top5 = (np.argsort(-probs, 1)[:, :k] == y[:, None]).any(1).mean()

    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    present = sorted(set(y.tolist()) | set(pred.tolist()))
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    in_test = np.array([s > 0 for s in S])
    macro = dict(precision=float(P[in_test].mean()), recall=float(R[in_test].mean()), f1=float(F[in_test].mean()))
    wP, wR, wF, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)

    rng = np.random.default_rng(0)
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(args.bootstrap)]
    ece, rel_rows = ece_score(conf, correct)

    metrics = {
        "source": source, "checkpoint": str(args.ckpt), "arch": args.arch or "full-model",
        "n_images": int(len(y)), "n_classes_in_eval": int(in_test.sum()), "n_classes_model": len(classes),
        "top1": float(correct.mean()), "top1_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        f"top{k}": float(top5),
        "macro": macro,
        "weighted": dict(precision=float(wP), recall=float(wR), f1=float(wF)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "ece_15bins": ece,
        "mean_confidence": float(conf.mean()),
        "eval_seconds": round(time.time() - t0, 1),
        "img_size": args.img_size, "norm": args.norm, "resize_mode": args.resize_mode,
    }
    if not args.no_latency:
        med, p90 = latency_ms(model, args.img_size)
        metrics["cpu_latency_ms_batch1"] = {"median": round(med, 2), "p90": round(p90, 2),
                                            "threads": torch.get_num_threads()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c, p_, r_, f_, s_ in sorted(zip(present, P, R, F, S), key=lambda t: t[3]):
            w.writerow([classes[c], f"{p_:.4f}", f"{r_:.4f}", f"{f_:.4f}", int(s_)])

    pairs = Counter((int(a), int(b)) for a, b in zip(y, pred) if a != b)
    with (out / "top_confusions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count"])
        for (a, b), n in pairs.most_common(50):
            w.writerow([classes[a], classes[b], n])

    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for (p_, _), t, pr, c in zip(ds.items, y, pred, conf):
            w.writerow([p_, classes[t], classes[pr], f"{c:.4f}", int(t == pr)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_in = [i for i in present if (y == i).any()]
    cm = confusion_matrix(y, pred, labels=labels_in, normalize="true")
    n = len(labels_in)
    fig_sz = min(18, max(5, n * 0.45))
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    if n <= 40:
        names = [classes[i].replace("___", " ").replace("__", " ").replace("_", " ") for i in labels_in]
        ax.set_xticks(range(n), names, rotation=90, fontsize=7)
        ax.set_yticks(range(n), names, fontsize=7)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    if cm[i, j] >= 0.01:
                        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if cm[i, j] > 0.5 else "black")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized confusion matrix ({n} classes, n={len(y)})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if rel_rows:
        mids = [(a + b) / 2 for a, b, *_ in rel_rows]
        ax.bar(mids, [r[3] for r in rel_rows], width=1 / 15, edgecolor="k", alpha=0.8, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=200)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"[done] results in {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
