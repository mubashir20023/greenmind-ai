# app/identify.py
from __future__ import annotations
import os
import json
from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import timm

# Optional detector (preferred)
try:
    from app.detector import DETECTOR
except Exception:
    DETECTOR = None

# -------- Config --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parents[1]

# Species classifier bits
MODEL_NAME   = os.getenv("MODEL_NAME", "resnet18")  # 'resnet18' | 'vit_base_patch16_224'
CKPT_PATH    = Path(os.getenv("CKPT_PATH", str(BASE_DIR / "models" / "checkpoints" / "plant_id_resnet18_best.pth")))
IDX2_PATH    = Path(os.getenv("IDX2_PATH",   str(BASE_DIR / "models" / "class_maps" / "idx_to_species.json")))
TOPK_DEFAULT = int(os.getenv("TOPK", "3"))

# Transforms must match your train/val setup
VAL_T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# -------- Class map --------
@lru_cache(maxsize=1)
def _load_class_map() -> Dict[int, str]:
    if not IDX2_PATH.exists():
        raise FileNotFoundError(f"Missing class map: {IDX2_PATH}")
    with IDX2_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# -------- Classifier model --------
@lru_cache(maxsize=1)
def _load_model():
    idx2 = _load_class_map()
    num_classes = len(idx2)
    if MODEL_NAME == "resnet18":
        m = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    elif MODEL_NAME == "vit_base_patch16_224":
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    else:
        raise RuntimeError(f"Unknown MODEL_NAME={MODEL_NAME}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    state = torch.load(CKPT_PATH, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval().to(DEVICE)
    return m

# -------- Helpers --------
@torch.inference_mode()
def _classify_pil(img: Image.Image, topk: int) -> List[Dict]:
    m = _load_model()
    idx2 = _load_class_map()
    x = VAL_T(ImageOps.exif_transpose(img).convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = m(x)                     # [1, C]
    probs  = F.softmax(logits, dim=1)[0]
    k = max(1, min(topk, probs.numel()))
    vals, idxs = torch.topk(probs, k=k, dim=0)
    return [{"label": idx2[int(i)], "score": float(p)} for p, i in zip(vals.tolist(), idxs.tolist())]

def _aggregate_max(votes: Dict[str, float], preds: List[Dict]):
    """Keep the max prob per species label."""
    for pr in preds:
        lbl, sc = pr["label"], float(pr["score"])
        if sc > votes.get(lbl, 0.0):
            votes[lbl] = sc

# -------- Public API --------
@torch.inference_mode()
def identify(img: Image.Image, gallery_root=None, topk: int = None) -> Tuple[Dict, List[Dict]]:
    """
    1) Use YOLO detector to crop {tree, leaf, flower}; classify crops
    2) If detector missing or no detections -> classify full image
    Returns: (best, alternatives) where each item is {'label': str, 'score': float}
    """
    topk = topk or TOPK_DEFAULT
    votes: Dict[str, float] = {}

    # Try detector
    crops: List[Image.Image] = []
    if DETECTOR and DETECTOR.is_available():
        try:
            dets = DETECTOR.detect(img)
            crops = [d["crop"] for d in dets if "crop" in d]
        except Exception:
            crops = []

    if crops:
        for crop in crops:
            _aggregate_max(votes, _classify_pil(crop, topk=topk))
    else:
        _aggregate_max(votes, _classify_pil(img, topk=topk))

    items = sorted(votes.items(), key=lambda kv: kv[1], reverse=True) or [("Unknown", 0.0)]
    alts = [{"label": k, "score": float(v)} for k, v in items[:topk]]
    return alts[0], alts
