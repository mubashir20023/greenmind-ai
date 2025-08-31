from __future__ import annotations
import os, json
from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import timm

# Optional detector (kept; harmless if weights absent)
try:
    from app.detector import DETECTOR
except Exception:
    DETECTOR = None

# --- Two hosted backends (Plant.id, Pl@ntNet) ---
from app.backends import PlantIdBackend, PlantNetBackend, try_backends

# -------- Config --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parents[1]

# Local species classifier bits (fallback)
MODEL_NAME   = os.getenv("MODEL_NAME", "resnet18")
CKPT_PATH    = Path(os.getenv("CKPT_PATH", str(BASE_DIR / "models" / "checkpoints" / "plant_id_resnet18_best.pth")))
IDX2_PATH    = Path(os.getenv("IDX2_PATH",   str(BASE_DIR / "models" / "class_maps" / "idx_to_species.json")))
with IDX2_PATH.open("r", encoding="utf-8") as f:
    IDX2 = json.load(f)
N_CLASSES = len(IDX2)

_MODEL = None

TOPK_DEFAULT = int(os.getenv("TOPK", "3"))
ORGAN_HINT = os.getenv("ORGAN_HINT", "").strip() or None  # optional: leaf|flower|fruit

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

# -------- Local classifier model --------
@lru_cache(maxsize=1)
def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_name = os.getenv("MODEL_NAME", "resnet18")
    ckpt_path = Path(os.getenv("CKPT_PATH", "models/checkpoints/plant_id_resnet18_best.pth"))

    model = timm.create_model(model_name, pretrained=True, num_classes=N_CLASSES)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[model] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[warn] checkpoint not found at {ckpt_path}; using ImageNet-pretrained backbone")

    model.eval().to(DEVICE)
    _MODEL = model
    return _MODEL

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

# -------- Public API --------
@torch.inference_mode()
def identify(img: Image.Image, gallery_root=None, topk: int = None) -> Tuple[Dict, List[Dict]]:
    """
    Tries hosted backends in priority order (top-2):
      1) Plant.id API (if PLANT_ID_API_KEY present)
      2) Pl@ntNet API (if PLANTNET_API_KEY present)
    Falls back to local classifier if both unavailable or fail.
    Returns: (best, alternatives) where each item is {label: str, score: float}
    """
    topk = topk or TOPK_DEFAULT

    # Optional crop list (ignored unless your YOLO weights are present)
    crops: List[Image.Image] = []
    if DETECTOR and DETECTOR.is_available():
        try:
            dets = DETECTOR.detect(img)
            crops = [d["crop"] for d in dets if "crop" in d]
        except Exception:
            crops = []

    # Prefer full image to avoid extra API calls; only use first crop if detector exists.
    query_image: Image.Image = (crops[0] if crops else img)

    # Try hosted backends
    be_list = [PlantIdBackend(), PlantNetBackend()]
    from app.backends import try_backends as _try
    be_name, be_best, be_alts = _try(query_image, be_list, topk=topk, organ_hint=ORGAN_HINT)
    if be_name != "none":
        os.environ["MODEL_NAME"] = be_name
        return be_best, be_alts

    # Fallback: local classifier
    preds = _classify_pil(query_image, topk=topk)
    os.environ["MODEL_NAME"] = "local"
    return preds[0], preds
