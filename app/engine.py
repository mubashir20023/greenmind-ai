# -*- coding: utf-8 -*-
"""
Inference engine selection for GreenMind.

GREENMIND_ENGINE = torch | onnx | auto (default)
  torch : PyTorch + timm + Ultralytics (full version, Grad-CAM; needs ~3-4 GB RAM)
  onnx  : ONNX Runtime only, no PyTorch at run time (lite version, fits 512 MB free hosting)
  auto  : torch if it can be imported, otherwise onnx

The ONNX models are produced by tools/export_onnx.py and live in models/onnx/ (override with ONNX_DIR):
  species_effb0.onnx  + species_effb0.json  + species_effb0_fc.npy      species classifier
  disease_effb0.onnx  + disease_effb0.json  + disease_effb0_fc.npy      leaf-disease classifier
  organ_yolo.onnx     + organ_yolo.json                                 tree / leaf / flower detector
  leaf_yolo.onnx      + leaf_yolo.json                                  leaf detector

Classifiers are exported with two outputs (logits, last feature map), so class activation maps (CAM)
and Eigen-CAM can be computed with NumPy, without gradients.
Pre-processing reproduces the torchvision transforms used in training exactly (PIL bilinear resize,
rounding of torchvision's CenterCrop), so ONNX and PyTorch see the same pixels.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parents[1]
ONNX_DIR = Path(os.getenv("ONNX_DIR", str(BASE_DIR / "models" / "onnx")))

NORMS = {
    "imagenet": (np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)),
    "half": (np.array([0.5, 0.5, 0.5], np.float32), np.array([0.5, 0.5, 0.5], np.float32)),
}


def _torch_importable() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _select_engine() -> str:
    e = os.getenv("GREENMIND_ENGINE", "auto").strip().lower()
    if e in ("torch", "onnx"):
        return e
    return "torch" if _torch_importable() else "onnx"


ENGINE = _select_engine()


def is_onnx() -> bool:
    return ENGINE == "onnx"


# --------------------------------------------------------------------------- runtime
def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # keep the resident memory small (important on 512 MB hosts)
    so.enable_cpu_mem_arena = os.getenv("ORT_MEM_ARENA", "0") == "1"
    so.enable_mem_pattern = False
    n = int(os.getenv("ORT_THREADS", "0"))
    if n > 0:
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def preprocess(pil: Image.Image, size: int = 224, resize_mode: str = "crop", norm: str = "imagenet") -> np.ndarray:
    """Same pixels as the torchvision eval transforms in tools/gm_data.py."""
    im = ImageOps.exif_transpose(pil).convert("RGB")
    if resize_mode == "crop":
        short = int(round(size * 256 / 224))
        w, h = im.size
        if w <= h:
            nw, nh = short, int(short * h / w)
        else:
            nh, nw = short, int(short * w / h)
        im = im.resize((nw, nh), Image.BILINEAR)
        left = int(round((nw - size) / 2.0))
        top = int(round((nh - size) / 2.0))
        im = im.crop((left, top, left + size, top + size))
    else:  # "square"
        im = im.resize((size, size), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    mean, std = NORMS[norm]
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def _normalize_map(m):
    m = np.maximum(m, 0)
    mx = m.max()
    return (m / mx) if mx > 0 else m


# --------------------------------------------------------------------------- classifiers
class OnnxClassifier:
    def __init__(self, stem: str):
        meta = json.loads((ONNX_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.classes = list(meta["classes"])
        self.size = int(meta.get("img_size", 224))
        self.resize_mode = meta.get("resize_mode", "crop")
        self.norm = meta.get("norm", "imagenet")
        self.fc_w = np.load(ONNX_DIR / meta["fc"]).astype(np.float32)  # (num_classes, channels)
        self.sess = _session(ONNX_DIR / f"{stem}.onnx")
        self.input_name = self.sess.get_inputs()[0].name

    def forward(self, pil):
        x = preprocess(pil, self.size, self.resize_mode, self.norm)
        logits, feats = self.sess.run(None, {self.input_name: x})
        return logits[0], feats[0]  # (C,), (K, h, w)

    def predict(self, pil) -> np.ndarray:
        logits, _ = self.forward(pil)
        return softmax(logits)

    def cam(self, feats, class_idx) -> np.ndarray:
        """Class activation map (Zhou et al. 2016): classifier weights x last feature map."""
        return _normalize_map(np.tensordot(self.fc_w[class_idx], feats, axes=(0, 0)))

    @staticmethod
    def eigen_cam(feats) -> np.ndarray:
        """Eigen-CAM: projection of the activations on their first principal component."""
        k, h, w = feats.shape
        a = feats.reshape(k, -1).T
        a = a - a.mean(axis=0)
        _, _, vt = np.linalg.svd(a, full_matrices=False)
        proj = (a @ vt[0]).reshape(h, w)
        if proj.max() < -proj.min():  # SVD sign is arbitrary
            proj = -proj
        return _normalize_map(proj)

    def predict_with_cam(self, pil, method: str = "cam"):
        logits, feats = self.forward(pil)
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        heat = self.eigen_cam(feats) if method == "eigen" else self.cam(feats, idx)
        return probs, idx, heat


# --------------------------------------------------------------------------- YOLO
def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


class OnnxYolo:
    """Ultralytics YOLOv8/11 detector exported to ONNX (output: 1 x (4+nc) x N)."""

    def __init__(self, stem: str):
        meta = json.loads((ONNX_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        self.names = {int(k): str(v).lower() for k, v in meta["names"].items()}
        self.sess = _session(ONNX_DIR / f"{stem}.onnx")
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(meta.get("imgsz") or inp.shape[2])

    def detect(self, pil, conf=0.25, iou=0.45, max_det=100):
        import cv2
        im = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
        H, W = im.shape[:2]
        s = self.imgsz
        r = min(s / H, s / W)
        nw, nh = int(round(W * r)), int(round(H * r))
        dw, dh = (s - nw) / 2, (s - nh) / 2
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (W, H) else im
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        p = self.sess.run(None, {self.input_name: x})[0][0].T  # (N, 4+nc)
        cls_scores = p[:, 4:]
        cls = cls_scores.argmax(1)
        sc = cls_scores[np.arange(len(cls)), cls]
        m = sc >= conf
        if not m.any():
            return []
        b, sc, cls = p[m, :4], sc[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        keep = _nms(xyxy + cls[:, None] * 7680.0, sc, iou)[:max_det]  # class-aware NMS
        return [(int(cls[i]), self.names.get(int(cls[i]), str(int(cls[i]))), float(sc[i]),
                 tuple(float(v) for v in xyxy[i])) for i in keep]


# --------------------------------------------------------------------------- lazy singletons
def _exists(stem):
    return (ONNX_DIR / f"{stem}.onnx").exists() and (ONNX_DIR / f"{stem}.json").exists()


@lru_cache(maxsize=1)
def species_model():
    return OnnxClassifier("species_effb0") if _exists("species_effb0") else None


@lru_cache(maxsize=1)
def disease_model():
    return OnnxClassifier("disease_effb0") if _exists("disease_effb0") else None


@lru_cache(maxsize=1)
def organ_detector():
    return OnnxYolo("organ_yolo") if _exists("organ_yolo") else None


@lru_cache(maxsize=1)
def leaf_detector():
    return OnnxYolo("leaf_yolo") if _exists("leaf_yolo") else None
