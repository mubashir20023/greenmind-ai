#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare GreenMind's two inference engines (PyTorch vs ONNX Runtime).

  accuracy : evaluate the ONNX classifiers of app/engine.py on an image folder (one sub-folder per class)
             and write metrics.json + predictions.csv in the same format as tools/evaluate.py, so the
             per-image predictions of both engines can be compared.
  resources: in a fresh process, load all four models of one engine, run them, and report resident memory
             (RSS) and median CPU latency per model. Run once per engine.
  packages : installed size of the Python packages each engine needs.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def _peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # Linux: kB


# --------------------------------------------------------------------------- accuracy
def cmd_accuracy(a):
    import numpy as np
    from PIL import Image
    from gm_data import read_imagefolder
    os.environ["GREENMIND_ENGINE"] = "onnx"
    from app import engine
    clf = engine.species_model() if a.task == "species" else engine.disease_model()
    assert clf is not None, f"ONNX {a.task} model not found in {engine.ONNX_DIR}"
    rows = read_imagefolder(a.imagefolder)
    if a.label_map:
        lm = json.loads(Path(a.label_map).read_text(encoding="utf-8"))
        rows = [(p, lm.get(l, l)) for p, l in rows]
    idx = {c: i for i, c in enumerate(clf.classes)}
    rows = [(p, l) for p, l in rows if l in idx]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = c1 = c5 = 0
    t0 = time.time()
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for i, (p, l) in enumerate(rows):
            with Image.open(p) as im:
                probs = clf.predict(im)
            top = np.argsort(-probs)[:5]
            y = idx[l]
            n += 1
            c1 += int(top[0] == y)
            c5 += int(y in top)
            w.writerow([p, l, clf.classes[top[0]], f"{probs[top[0]]:.4f}", int(top[0] == y)])
            if i % 2000 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    m = {"engine": "onnx", "task": a.task, "n_images": n, "top1": c1 / max(1, n), "top5": c5 / max(1, n),
         "seconds": round(time.time() - t0, 1), "source": a.imagefolder}
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))


# --------------------------------------------------------------------------- resources
def cmd_resources(a):
    import numpy as np
    from PIL import Image
    img = Image.open(a.image).convert("RGB")
    base = _rss_mb()
    lat = {}

    def bench(name, fn):
        fn()  # warm-up
        ts = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        lat[name] = round(float(np.median(ts)), 1)

    if a.engine == "onnx":
        os.environ["GREENMIND_ENGINE"] = "onnx"
        os.environ["ORT_THREADS"] = str(a.threads)
        t0 = time.time()
        from app import engine
        sp, ds, og, lf = engine.species_model(), engine.disease_model(), engine.organ_detector(), engine.leaf_detector()
        missing = [n for n, m in [("species", sp), ("disease", ds), ("organ_yolo", og), ("leaf_yolo", lf)] if m is None]
        assert not missing, f"ONNX models missing in {engine.ONNX_DIR}: {missing}"
        load_s = time.time() - t0
        after_load = _rss_mb()
        bench("species", lambda: sp.predict_with_cam(img))
        bench("disease", lambda: ds.predict_with_cam(img))
        bench("organ_detector", lambda: og.detect(img))
        bench("leaf_detector", lambda: lf.detect(img))
    else:
        import torch
        import timm
        from gm_data import build_transforms, load_classes
        torch.set_num_threads(a.threads)
        t0 = time.time()
        from ultralytics import YOLO

        def clf(ckpt, classes, norm, mode):
            m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(load_classes(classes)))
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            return m.eval(), build_transforms(224, norm, False, mode)

        sp, spt = clf(a.species_ckpt, a.species_classes, "imagenet", "crop")
        ds, dst = clf(a.disease_ckpt, a.disease_classes, "half", "square")
        og, lf = YOLO(a.organ_yolo), YOLO(a.leaf_yolo)
        load_s = time.time() - t0
        after_load = _rss_mb()

        def run(m, t):
            with torch.inference_mode():
                return m(t(img).unsqueeze(0))
        bench("species", lambda: run(sp, spt))
        bench("disease", lambda: run(ds, dst))
        bench("organ_detector", lambda: og.predict(img, verbose=False))
        bench("leaf_detector", lambda: lf.predict(img, verbose=False))
    res = {"engine": a.engine, "threads": a.threads, "rss_after_imports_mb": round(base, 1),
           "rss_after_loading_models_mb": round(after_load, 1), "rss_after_inference_mb": round(_rss_mb(), 1),
           "peak_rss_mb": round(_peak_mb(), 1), "model_load_seconds": round(load_s, 2),
           "median_latency_ms": lat}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- packages
def cmd_packages(a):
    import importlib.util

    def size(pkg):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            return None
        tot = 0
        for root in spec.submodule_search_locations:
            for dp, _, fs in os.walk(root):
                tot += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
        return round(tot / 1e6, 1)
    stacks = {"torch": ["torch", "torchvision", "timm", "ultralytics", "pytorch_grad_cam", "cv2"],
              "onnx": ["onnxruntime", "cv2"]}
    res = {k: {p: size(p) for p in v} for k, v in stacks.items()}
    for k in res:
        res[k]["total_mb"] = round(sum(v for v in res[k].values() if v), 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("accuracy")
    s.add_argument("--task", choices=["species", "disease"], required=True)
    s.add_argument("--imagefolder", required=True)
    s.add_argument("--label_map", default="")
    s.add_argument("--out", required=True)
    r = sub.add_parser("resources")
    r.add_argument("--engine", choices=["torch", "onnx"], required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--repeats", type=int, default=20)
    for k in ("species_ckpt", "species_classes", "disease_ckpt", "disease_classes", "organ_yolo", "leaf_yolo"):
        r.add_argument(f"--{k}", default="")
    r.add_argument("--out", required=True)
    p = sub.add_parser("packages")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"accuracy": cmd_accuracy, "resources": cmd_resources, "packages": cmd_packages}[a.cmd](a)


if __name__ == "__main__":
    main()
