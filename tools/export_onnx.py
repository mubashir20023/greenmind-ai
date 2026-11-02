#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export GreenMind's PyTorch models to ONNX for the lite engine (app/engine.py).

  python tools/export_onnx.py \
      --species_ckpt models/checkpoints/plant_id_effb0_best.pth \
      --species_classes models/class_maps/idx_to_species_effb0.json \
      --disease_ckpt models/health/leaf_health_effb0_field.pth \
      --disease_classes models/health/classes.json \
      --organ_yolo runs/detect/train/weights/best.pt \
      --leaf_yolo models/health/yolo_leaf.pt \
      --out models/onnx

Classifiers get two outputs (logits, last feature map) plus the classifier weight matrix (.npy) so that
CAM / Eigen-CAM can be computed without gradients. Needs torch, timm, ultralytics and onnx.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm_data import load_classes  # noqa: E402


class LogitsAndFeatures(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.forward_features(x)
        return self.m.forward_head(f), f


def export_classifier(ckpt, classes_path, arch, out, stem, norm, resize_mode, img_size=224):
    import timm
    classes = load_classes(classes_path)
    m = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        m = state
    else:
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        m.load_state_dict(state, strict=True)
    m.eval()
    w = LogitsAndFeatures(m).eval()
    x = torch.randn(1, 3, img_size, img_size)
    path = out / f"{stem}.onnx"
    kw = dict(input_names=["input"], output_names=["logits", "features"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
              opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(w, x, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(w, x, str(path), **kw)
    fc = m.get_classifier()
    np.save(out / f"{stem}_fc.npy", fc.weight.detach().cpu().numpy().astype(np.float32))
    meta = {"arch": arch, "img_size": img_size, "resize_mode": resize_mode, "norm": norm,
            "classes": classes, "fc": f"{stem}_fc.npy", "source_checkpoint": str(ckpt)}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    # numerical check against PyTorch
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, img_size, img_size)
    with torch.inference_mode():
        lt, ft = w(x)
    lo, fo = s.run(None, {"input": x.numpy()})
    d = {"logits_max_abs_diff": float(np.abs(lt.numpy() - lo).max()),
         "features_max_abs_diff": float(np.abs(ft.numpy() - fo).max()),
         "argmax_agree": float((lt.numpy().argmax(1) == lo.argmax(1)).mean())}
    print(stem, d)
    return d


def _legacy_onnx_exporter():
    """PyTorch >= 2.9 defaults to the dynamo exporter (needs onnxscript, opset >= 18). Ultralytics calls
    torch.onnx.export without choosing, so force the TorchScript exporter, which supports opset 12."""
    orig = torch.onnx.export
    if getattr(orig, "_gm_legacy", False):
        return

    def export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        try:
            return orig(*args, **kwargs)
        except TypeError:  # torch without the dynamo argument
            kwargs.pop("dynamo", None)
            return orig(*args, **kwargs)
    export._gm_legacy = True
    torch.onnx.export = export


def export_yolo(pt, out, stem, imgsz=640):
    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # don't let Ultralytics pip-install during export
    _legacy_onnx_exporter()
    from ultralytics import YOLO
    # Ultralytics writes the .onnx next to the .pt, so work on a copy in a writable folder
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    local_pt = work / f"{stem}.pt"
    shutil.copy(str(pt), str(local_pt))
    y = YOLO(str(local_pt))
    p = y.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    shutil.move(str(p), str(out / f"{stem}.onnx"))
    meta = {"names": {int(k): v for k, v in dict(y.names).items()}, "imgsz": imgsz, "source_checkpoint": str(pt)}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(stem, meta["names"])
    res = yolo_parity(y, out, stem)
    shutil.rmtree(work, ignore_errors=True)
    return res


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def yolo_parity(y, out, stem, check_dir="/kaggle/working/PlantDoc-Dataset/test", n=60, conf=0.25):
    """Ultralytics (PyTorch) vs app/engine.OnnxYolo on real photos: box count and IoU of matched boxes."""
    import os
    from PIL import Image
    imgs = sorted(p for p in Path(check_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n] \
        if Path(check_dir).exists() else []
    if not imgs:
        return {}
    os.environ["ONNX_DIR"] = str(out)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.engine import OnnxYolo
    ox = OnnxYolo(stem)
    same_count, ious, n_t, n_o = 0, [], 0, 0
    for p in imgs:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        r = y.predict(im, conf=conf, verbose=False)[0]
        tb = [(int(c), b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        ob = [(c, b) for c, _, _, b in ox.detect(im, conf=conf)]
        n_t += len(tb)
        n_o += len(ob)
        same_count += int(len(tb) == len(ob))
        for c, b in tb:
            cand = [_iou(b, b2) for c2, b2 in ob if c2 == c]
            ious.append(max(cand) if cand else 0.0)
    res = {"images": len(imgs), "boxes_torch": n_t, "boxes_onnx": n_o,
           "images_same_box_count": same_count,
           "mean_best_iou": round(float(np.mean(ious)), 4) if ious else None,
           "share_boxes_iou_ge_0.9": round(float(np.mean(np.array(ious) >= 0.9)), 4) if ious else None}
    print(stem, "parity", res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species_ckpt")
    ap.add_argument("--species_classes")
    ap.add_argument("--species_arch", default="efficientnet_b0")
    ap.add_argument("--disease_ckpt")
    ap.add_argument("--disease_classes")
    ap.add_argument("--disease_arch", default="efficientnet_b0")
    ap.add_argument("--organ_yolo")
    ap.add_argument("--leaf_yolo")
    ap.add_argument("--out", default="models/onnx")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    if a.species_ckpt:
        report["species_effb0"] = export_classifier(a.species_ckpt, a.species_classes, a.species_arch,
                                                    out, "species_effb0", "imagenet", "crop")
    if a.disease_ckpt:
        report["disease_effb0"] = export_classifier(a.disease_ckpt, a.disease_classes, a.disease_arch,
                                                    out, "disease_effb0", "half", "square")
    if a.organ_yolo:
        report["organ_yolo"] = export_yolo(a.organ_yolo, out, "organ_yolo")
    if a.leaf_yolo:
        report["leaf_yolo"] = export_yolo(a.leaf_yolo, out, "leaf_yolo")
    for f in sorted(out.iterdir()):
        report.setdefault("files_mb", {})[f.name] = round(f.stat().st_size / 1e6, 2)
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
