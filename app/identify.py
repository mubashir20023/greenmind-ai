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

# Optional YOLO detector – safe if weights absent
try:
    from app.detector import DETECTOR
except Exception:
    DETECTOR = None

# Two hosted backends (only used if API keys exist)
from app.backends import PlantIdBackend, PlantNetBackend

# ---------------- Config ----------------
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parents[1]

# Local classifier bits (fallback)
CKPT_PATH = Path(os.getenv("CKPT_PATH", str(BASE_DIR / "models" / "checkpoints" / "plant_id_resnet18_best.pth")))
IDX2_PATH = Path(os.getenv("IDX2_PATH",  str(BASE_DIR / "models" / "class_maps" / "idx_to_species.json")))

TOPK_DEFAULT = int(os.getenv("TOPK", "3"))

# Optional static hint (used if YOLO can’t infer organ)
ORGAN_HINT_ENV = (os.getenv("ORGAN_HINT", "") or "").strip() or None

# YOLO crop selection for the ensemble
YOLO_CONF           = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_MIN_AREA_FRAC  = float(os.getenv("YOLO_MIN_AREA_FRAC", "0.04"))
ENSEMBLE_MAX_CROPS  = int(os.getenv("ENSEMBLE_MAX_CROPS", "3"))

# Source weights for fusion
SRC_WEIGHTS = {
    "plant.id": float(os.getenv("W_PLANTID",  "1.0")),
    "plantnet": float(os.getenv("W_PLANTNET", "1.0")),
    "local":    float(os.getenv("W_LOCAL",    "0.6")),
}

# Inference transforms (must match training normalization)
VAL_T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ---------------- Class map & model (lazy) ----------------
@lru_cache(maxsize=1)
def _load_class_map() -> Dict[int, str]:
    """
    idx -> species label, loaded lazily.
    Returns {} if file is missing (local classifier will be skipped).
    """
    try:
        with IDX2_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # keys may be strings; convert to int
        return {int(k): str(v) for k, v in raw.items()}
    except Exception:
        return {}

@lru_cache(maxsize=1)
def _load_model():
    """
    Create model only if class map is available; otherwise return None.
    """
    idx2 = _load_class_map()
    if not idx2:
        print(f"[local] class map not found at {IDX2_PATH}; skipping local classifier")
        return None

    num_classes = len(idx2)
    model_name  = os.getenv("MODEL_NAME", "resnet18")
    ckpt_path   = Path(os.getenv("CKPT_PATH", str(CKPT_PATH)))

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[model] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[warn] checkpoint not found at {ckpt_path}; using ImageNet-pretrained backbone")

    model.eval().to(DEVICE)
    return model


# ---------------- Local classify ----------------
@torch.inference_mode()
def _classify_pil(img: Image.Image, topk: int) -> List[Dict]:
    m = _load_model()
    if m is None:
        return []  # local disabled if no class map

    idx2 = _load_class_map()
    x = VAL_T(ImageOps.exif_transpose(img).convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = m(x)                     # [1, C]
    probs  = F.softmax(logits, dim=1)[0]
    k = max(1, min(topk, probs.numel()))
    vals, idxs = torch.topk(probs, k=k, dim=0)

    out = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        label = idx2.get(int(i), f"class_{int(i)}")
        out.append({"label": label, "score": float(p)})
    return out


# ---------------- Helpers ----------------
def _normalize_label(s: str) -> str:
    return " ".join((s or "Unknown").strip().split())

def _make_candidates(img: Image.Image) -> Tuple[List[Image.Image], Optional[str]]:
    """
    Returns (candidates, organ_hint)
    candidates: [full image] + up to ENSEMBLE_MAX_CROPS YOLO crops passing gates
    organ_hint: inferred from top detection ('leaf'/'flower'/'fruit') if present, else ORGAN_HINT_ENV
    """
    candidates: List[Image.Image] = [img]
    organ_hint: Optional[str] = ORGAN_HINT_ENV

    if DETECTOR and DETECTOR.is_available():
        W, H = img.size
        try:
            dets = DETECTOR.detect(img)  # [{'cls','conf','box','crop'}]
        except Exception:
            dets = []

        # filter by confidence and min box area
        valid = []
        for d in dets:
            (x1, y1, x2, y2) = d.get("box", (0,0,0,0))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            frac = (area / float(W * H)) if (W * H) else 0.0
            if float(d.get("conf", 0.0)) >= YOLO_CONF and frac >= YOLO_MIN_AREA_FRAC:
                valid.append(d)

        # pick up to N largest/confident crops
        valid.sort(
            key=lambda d: (float(d.get("conf", 0.0)),
                           (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1])),
            reverse=True
        )
        for d in valid[:max(0, ENSEMBLE_MAX_CROPS)]:
            cimg = d.get("crop")
            if isinstance(cimg, Image.Image):
                candidates.append(cimg)

        # infer organ hint from the most confident detection (prefer flower/fruit over leaf if tie)
        if valid:
            def _rank(d):
                cls = (d.get("cls") or "").lower()
                pri = 2 if cls == "flower" else (1 if cls == "fruit" else 0)
                return (pri, float(d.get("conf", 0.0)))
            top = sorted(valid, key=_rank, reverse=True)[0]
            cls = (top.get("cls") or "").lower()
            if cls in ("leaf", "flower", "fruit"):
                organ_hint = cls

    return candidates, organ_hint


def _max_per_source(votes: List[Tuple[str, List[Dict]]]) -> Dict[str, Dict[str, float]]:
    """
    Collapse multiple candidates per source by taking max score per label.
    Returns: {source: {label: score}}
    """
    out: Dict[str, Dict[str, float]] = {}
    for src, alts in votes:
        bucket = out.setdefault(src, {})
        for a in (alts or []):
            lbl = _normalize_label(a.get("label", "Unknown"))
            sc  = float(a.get("score", 0.0))
            bucket[lbl] = max(bucket.get(lbl, 0.0), sc)
    return out


def _fuse_sources(src_maps: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Weighted sum across sources (per label), then normalized to probabilities.
    """
    pool = set()
    for m in src_maps.values():
        pool.update(m.keys())

    fused: Dict[str, float] = {}
    for lbl in pool:
        val = 0.0
        for src, smap in src_maps.items():
            w = float(SRC_WEIGHTS.get(src, 1.0))
            val += w * float(smap.get(lbl, 0.0))
        fused[lbl] = val

    # normalize
    s = sum(max(0.0, v) for v in fused.values()) or 1.0
    return {k: max(0.0, v)/s for k, v in fused.items()}


# ---------------- Public API ----------------
@torch.inference_mode()
def identify(img: Image.Image, gallery_root=None, topk: int = None) -> Tuple[Dict, List[Dict]]:
    """
    Ensemble identification:
      - Full image + up to N YOLO crops → call Plant.id + Pl@ntNet on each when available
      - Local classifier vote on full image (if class map exists)
      - Fuse per-source max scores with configurable weights; return (best, alternatives)
    NOTE: Non-plant/low-confidence gating is handled in app.main route, not here.
    """
    topk = topk or TOPK_DEFAULT

    candidates, organ_hint = _make_candidates(img)

    # Hosted backends (only used if keys present)
    be_plantid = PlantIdBackend()
    be_plantnet = PlantNetBackend()

    # Collect votes: list of (source_name, [ {label,score}, ... ])
    votes: List[Tuple[str, List[Dict]]] = []
    plantid_is_plant_probs: List[float] = []

    for cimg in candidates:
        if be_plantid.available():
            try:
                best, alts = be_plantid.identify(cimg, topk=topk, organ_hint=organ_hint)
                votes.append(("plant.id", alts))
                isp = best.get("is_plant_prob")
                if isinstance(isp, (int, float)):
                    plantid_is_plant_probs.append(float(isp))
            except Exception:
                pass
        if be_plantnet.available():
            try:
                best, alts = be_plantnet.identify(cimg, topk=topk, organ_hint=organ_hint)
                votes.append(("plantnet", alts))
            except Exception:
                pass

    # Local vote (always try; it’s cheap)
    try:
        local_alts = _classify_pil(candidates[0], topk=topk)
        if local_alts:
            votes.append(("local", local_alts))
    except Exception:
        pass

    if not votes:
        # Nothing succeeded — stay graceful
        os.environ["MODEL_NAME"] = "local"
        return {"label": "Unknown", "score": 0.0}, [{"label": "Unknown", "score": 0.0}]

    # Per-source max, then fuse with weights
    by_source = _max_per_source(votes)
    fused = _fuse_sources(by_source)

    # Pick best + build alt list
    best_label = max(fused.items(), key=lambda kv: kv[1])[0]
    best_score = float(fused[best_label])
    alts = sorted(
        [{"label": lbl, "score": float(sc)} for lbl, sc in fused.items()],
        key=lambda x: x["score"],
        reverse=True
    )[:max(3, topk)]

    best = {"label": best_label, "score": best_score}
    if plantid_is_plant_probs:
        best["is_plant_prob"] = sum(plantid_is_plant_probs) / len(plantid_is_plant_probs)

    # For UI badge: which source contributed most (weighted) to the best label?
    contribs = {}
    for src, smap in by_source.items():
        contribs[src] = float(SRC_WEIGHTS.get(src, 1.0)) * float(smap.get(best_label, 0.0))
    winner_src = max(contribs.items(), key=lambda kv: kv[1])[0] if contribs else "local"
    os.environ["MODEL_NAME"] = winner_src

    return best, alts
