# app/health.py
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image  # for BGR -> PIL conversion

from .engine import is_onnx
from . import engine as onnx_engine

if is_onnx():  # lite engine: ONNX Runtime only, no PyTorch
    torch = F = YOLO = None
    HAVE_CAM = False
else:
    import torch
    import torch.nn.functional as F
    # Optional Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except Exception:
        YOLO = None
    # Grad-CAM for classifiers
    try:
        from pytorch_grad_cam import GradCAM, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        HAVE_CAM = True
    except Exception:
        HAVE_CAM = False

# Optional Plant.id health (provided in app/backends.py)
try:
    from .backends import health_assess as plantid_health_assess
except Exception:
    plantid_health_assess = None

# Plant identification (identify.py)
try:
    # identify(img: PIL.Image, topk=None) -> (best: dict, alts: list[dict])
    from .identify import identify as identify_plant
except Exception:
    identify_plant = None

from pathlib import Path
_HEALTH_DIR = Path(__file__).resolve().parent.parent  # project root
# ============================ Config ============================

@dataclass
class HealthConfig:
    """
    Overall health pipeline configuration controlled by env vars.

    Modes:
      - plantid    : use Plant.id only (no local models)
      - classifier : local classifier on full image (with CAM overlays)
      - detector   : YOLO-only; diseases are YOLO classes
      - hybrid     : YOLO crops -> local classifier per crop (recommended)
    """
    # hybrid (leaf crops) is the default when a leaf detector is installed: +3 points on PlantDoc (paper, Table 3)
    mode: str = os.getenv("HEALTH_MODE", "auto").lower()

    topk: int = int(os.getenv("HEALTH_TOPK", 3))

    # Classifier
    # Field-trained EfficientNet-B0 (PlantVillage + PlantDoc-train, see paper / notebooks/04) when present,
    # otherwise the original full-model checkpoint.
    clf_weights: str = os.getenv(
        "HEALTH_CLF_WEIGHTS",
        "models/health/leaf_health_effb0_field.pth"
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "models/health/leaf_health_effb0_field.pth"))
        else "models/health/leaf_health_effb3.pt",
    )
    clf_arch: str = os.getenv("HEALTH_ARCH", "efficientnet_b0")  # used for state_dict checkpoints
    classes_path: str = os.getenv(
        "HEALTH_CLASSES",
        "models/health/classes.json",
    )
    device: str = os.getenv("HEALTH_DEVICE", "cpu")  # "cpu" or "cuda"
    input_size: int = int(os.getenv("HEALTH_INPUT_SIZE", 224))  # must match training size

    # YOLO leaf detector (trained on PlantDoc leaf boxes, notebooks/05) used for hybrid crops
    yolo_weights: str = os.getenv(
        "HEALTH_YOLO_WEIGHTS",
        "models/health/yolo_leaf.pt",
    )
    conf: float = float(os.getenv("HEALTH_CONF", 0.35))
    min_area_frac: float = float(os.getenv("HEALTH_MIN_AREA_FRAC", 0.02))

    # Plant.id fallback/merge
    use_plantid: bool = os.getenv("USE_PLANTID_HEALTH", "0") == "1"


CFG = HealthConfig()


def _leaf_detector_available() -> bool:
    if is_onnx():
        return onnx_engine.leaf_detector() is not None
    p = CFG.yolo_weights if os.path.isabs(CFG.yolo_weights) else str(_HEALTH_DIR / CFG.yolo_weights)
    return YOLO is not None and os.path.exists(p)


if CFG.mode == "auto":
    CFG.mode = "hybrid" if _leaf_detector_available() else "classifier"

# lazy singletons
_CLF_MODEL: Optional[Any] = None
_CLF_CLASSES: Optional[List[str]] = None
_YOLO_HEALTH = None


# ============================ Utils ============================

def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def _normalize_img_for_cam(img_rgb: np.ndarray) -> np.ndarray:
    arr = img_rgb.astype(np.float32) / 255.0
    return np.clip(arr, 0, 1)


def _softmax_topk(t, k: int) -> Tuple[List[float], List[int]]:
    prob = F.softmax(t, dim=1)[0]
    tk = torch.topk(prob, k=min(k, prob.size(0)))
    return tk.values.tolist(), tk.indices.tolist()


# ---------- Disease name helpers ----------

def _pretty_disease_name(raw: str) -> str:
    """
    Convert dataset-style labels like:

      - 'Pepper__bell___Bacterial_spot'
      - 'Tomato_Late_blight'
      - 'Tomato_Leaf_Mold'

    into human-friendly, crop-agnostic names like:

      - 'bacterial spot'
      - 'late blight'
      - 'leaf mold'
    """
    if not raw:
        return "unspecified leaf problem"

    label = raw.strip()
    lower = label.lower()

    # Healthy shortcuts
    if "healthy" in lower:
        return "healthy (no obvious disease)"

    # Pattern 1: PlantVillage 'Pepper__bell___Bacterial_spot'
    if "___" in label:
        _, disease = label.split("___", 1)
        disease = disease.replace("_", " ").strip()
        return disease or "leaf problem"

    # Pattern 2: 'Tomato_Late_blight', 'Tomato_Leaf_Mold', ...
    if "_" in label:
        _crop, disease = label.split("_", 1)
        disease = disease.replace("_", " ").strip()
        return disease or "leaf problem"

    # Fallback
    return label.replace("_", " ")


# ---------- Identification helper (identify.py) ----------

def _run_identification(image_bgr: np.ndarray, topk: Optional[int] = None) -> Dict[str, Any]:
    """
    Call identify.py FIRST to get plant name and candidates.

    Returns a dict we can merge into the health result:

    {
        "plant_name": str | None,
        "plant_confidence": float | None,
        "plant_candidates": [ {"name": str, "prob": float}, ... ],
        "identify_best": dict | None,
        "identify_topk": list | None,
    }
    """
    if identify_plant is None:
        return {
            "plant_name": None,
            "plant_confidence": None,
            "plant_candidates": [],
            "identify_best": None,
            "identify_topk": None,
        }

    try:
        # BGR (OpenCV) -> RGB -> PIL
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        best, alts = identify_plant(pil_img, topk=topk)

        plant_name = None
        plant_confidence = None
        if isinstance(best, dict):
            plant_name = best.get("label") or best.get("name")
            plant_confidence = float(best.get("score", best.get("prob", 0.0) or 0.0))

        candidates = []
        if isinstance(alts, list):
            for item in alts:
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("name")
                prob = float(item.get("score", item.get("prob", 0.0) or 0.0))
                if label is None:
                    continue
                candidates.append({"name": label, "prob": prob})

        return {
            "plant_name": plant_name,
            "plant_confidence": plant_confidence,
            "plant_candidates": candidates,
            "identify_best": best,
            "identify_topk": alts,
        }
    except Exception as e:
        # fail soft – never break health flow
        return {
            "plant_name": None,
            "plant_confidence": None,
            "plant_candidates": [],
            "identify_best": {"error": str(e)},
            "identify_topk": None,
        }


# ============================ Loads ============================

def load_yolo_health():
    """Leaf detector for the hybrid / detector modes (PyTorch engine). Returns None if unavailable."""
    global _YOLO_HEALTH
    if _YOLO_HEALTH is not None:
        return _YOLO_HEALTH
    p = CFG.yolo_weights if os.path.isabs(CFG.yolo_weights) else str(_HEALTH_DIR / CFG.yolo_weights)
    if YOLO is None or not os.path.exists(p):
        return None
    _YOLO_HEALTH = YOLO(p)
    return _YOLO_HEALTH


def load_classifier() -> Tuple[Optional[Any], List[str]]:
    global _CLF_MODEL, _CLF_CLASSES
    if _CLF_MODEL is not None:
        return _CLF_MODEL, _CLF_CLASSES or ["healthy", "diseased"]

    if is_onnx():
        m = onnx_engine.disease_model()
        if m is None:
            print("[health] ONNX disease model not found; returning unknown")
            return None, ["healthy", "diseased"]
        _CLF_MODEL, _CLF_CLASSES = m, list(m.classes)
        return _CLF_MODEL, _CLF_CLASSES

    # Resolve path relative to project root (fixes working-directory issues)
    weights_path = str(_HEALTH_DIR / CFG.clf_weights) if not os.path.isabs(CFG.clf_weights) else CFG.clf_weights
    classes_path = str(_HEALTH_DIR / CFG.classes_path) if not os.path.isabs(CFG.classes_path) else CFG.classes_path

    if not os.path.exists(weights_path):
        print(f"[health] classifier weights not found at {weights_path} — returning unknown")
        return None, ["healthy", "diseased"]

    try:
        map_loc = "cuda" if (CFG.device == "cuda" and torch.cuda.is_available()) else "cpu"

        # weights_only=False needed for full-model .pt files (PyTorch 2.x)
        loaded = torch.load(weights_path, map_location=map_loc, weights_only=False)

        # Handle both full-model and state-dict saves
        if isinstance(loaded, dict):
            state = loaded.get("model") or loaded.get("state_dict") or loaded
            if not os.path.exists(classes_path):
                raise RuntimeError(f"state_dict checkpoint needs a class list at {classes_path}")
            with open(classes_path, "r", encoding="utf-8") as f:
                n_classes = len(json.load(f))
            import timm
            model = timm.create_model(CFG.clf_arch, pretrained=False, num_classes=n_classes)
            model.load_state_dict(state, strict=True)
        else:
            model = loaded
        model.eval()
        if map_loc == "cuda":
            model = model.to("cuda")

    except Exception as e:
        print(f"[health] failed to load classifier: {e}")
        return None, ["healthy", "diseased"]

    # Load classes
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = ["healthy", "diseased"]

    _CLF_MODEL, _CLF_CLASSES = model, classes
    return _CLF_MODEL, _CLF_CLASSES

# ============================ CAM ============================

def _pick_target_layer(model):
    """Last convolution of the disease classifier (the spatial layer Grad-CAM needs)."""
    last = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def _blend(img_rgb_sq: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Heat map (any size, values 0..1) over the square model input; returns BGR for cv2.imwrite."""
    s = img_rgb_sq.shape[0]
    h = cv2.resize(heat.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap(np.uint8(255 * np.clip(h, 0, 1)), cv2.COLORMAP_JET)  # BGR
    base = cv2.cvtColor(img_rgb_sq, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 0.55, col, 0.45, 0)


def _generate_cam_overlays(model, img_rgb: np.ndarray, tensor, pred_idx: int):
    """
    Heat maps that explain the DISEASE classifier's own prediction (both engines):
      PyTorch: Grad-CAM (class-specific) + Eigen-CAM on the last convolution.
      ONNX   : CAM (class-specific, classifier weights x features) + Eigen-CAM.
    Returns [(name, overlay BGR image), ...].
    """
    s = int(CFG.input_size)
    img_sq = cv2.resize(img_rgb, (s, s), interpolation=cv2.INTER_AREA)
    if is_onnx():
        pil = Image.fromarray(img_rgb.astype("uint8"), mode="RGB")
        _, feats = model.forward(pil)
        return [("cam", _blend(img_sq, model.cam(feats, pred_idx))),
                ("eigencam", _blend(img_sq, model.eigen_cam(feats)))]
    try:
        if not HAVE_CAM:
            return []
        layer = _pick_target_layer(model)
        if layer is None:
            return []
        targets = [ClassifierOutputTarget(pred_idx)]
        with GradCAM(model=model, target_layers=[layer]) as gc:
            g = gc(input_tensor=tensor, targets=targets)[0]
        with EigenCAM(model=model, target_layers=[layer]) as ec:
            e = ec(input_tensor=tensor, targets=targets)[0]
        return [("gradcam", _blend(img_sq, g)), ("eigencam", _blend(img_sq, e))]
    except Exception as ex:
        print(f"[health] CAM failed: {ex}")
        return []

# ============================ Preprocess ============================

def _preprocess_for_clf(img_rgb: np.ndarray, size: Optional[int] = None):
    s = int(size or CFG.input_size)
    img = cv2.resize(img_rgb, (s, s), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # CHW
    t = torch.from_numpy(img).unsqueeze(0)
    if CFG.device == "cuda" and torch.cuda.is_available():
        t = t.to("cuda")
    return t


# ============================ Detection crops ============================

def _leaf_crops_or_full(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Use YOLO health detector (if configured) to get leaf/lesion crops.
    Fallback: full image as a single crop.
    """
    H, W = image_bgr.shape[:2]
    area = float(H * W) if (H and W) else 1.0
    results: List[Dict[str, Any]] = []

    yolo = None
    if CFG.mode in ("detector", "hybrid") and is_onnx():
        det = onnx_engine.leaf_detector()
        if det is not None:
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            for cls_id, _, conf, (x1, y1, x2, y2) in det.detect(pil, conf=CFG.conf):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if (x2 - x1) * (y2 - y1) / area < CFG.min_area_frac:
                    continue
                results.append({"box": [x1, y1, x2, y2], "crop": image_bgr[max(0, y1):y2, max(0, x1):x2],
                                "detector": {"cls": cls_id, "conf": conf}})
    try:
        if CFG.mode in ("detector", "hybrid") and not is_onnx():
            yolo = load_yolo_health()
    except Exception:
        yolo = None

    if yolo is not None:
        pred = yolo.predict(source=image_bgr, conf=CFG.conf, verbose=False)[0]
        for b in pred.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) / area < CFG.min_area_frac:
                continue
            crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
            cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
            conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
            results.append({
                "box": [x1, y1, x2, y2],
                "crop": crop,
                "detector": {"cls": cls_id, "conf": conf},
            })

    if not results:
        results = [{"box": [0, 0, W, H], "crop": image_bgr}]
    return results


# ============================ Plant.id only ============================

def _plantid_only(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Fast path when HEALTH_MODE=plantid (no local models)."""
    if plantid_health_assess is None:
        return {"status": "unknown", "confidence": 0.0, "diseases": [], "crops": []}

    ok, pid = plantid_health_assess(image_bgr)
    if not ok or not isinstance(pid, dict):
        return {"status": "unknown", "confidence": 0.0, "diseases": [], "crops": []}

    # pid["plantid"] = {"is_healthy": bool|None, "diseases": [{name, prob}, ...]}
    info = pid.get("plantid", {})
    is_healthy = info.get("is_healthy", None)

    diseases = info.get("diseases") or []
    # confidence heuristic: max prob if present, else 1.0 for healthy, else 0
    probs = [float(d.get("prob", 0.0)) for d in diseases if isinstance(d, dict)]
    if is_healthy is True:
        probs.append(1.0)
    conf = max(probs) if probs else (1.0 if is_healthy else 0.0)

    return {
        "status": "healthy" if is_healthy is True else ("diseased" if is_healthy is False or diseases else "unknown"),
        "confidence": float(conf),
        "diseases": diseases,
        "crops": [],
        "external": pid,
    }


# ============================ Main ============================

def assess_health(image_bgr: np.ndarray, out_dir: str = "runs/health") -> Dict[str, Any]:
    """
    Unified health assessment entrypoint.

    Returns a dict:

    {
      "plant_name": str | null,
      "plant_confidence": float | null,
      "plant_candidates": [ {"name": str, "prob": float}, ... ],
      "identify_best": {...} | null,
      "identify_topk": [... ] | null,

      "status": "healthy" | "diseased" | "unknown",
      "confidence": float,  # 0..1

      "diseases": [ {"name": str, "prob": float}, ... ],  # pretty names
      "crops": [
        {
          "box":[x1,y1,x2,y2],
          "label": str,          # raw dataset label
          "pretty_label": str,   # human-readable
          "prob": float,
          "topk":[{"name":str,"prob":float}, ...],  # pretty names
          "xai":[ "runs/health/crop1_gradcam.jpg", ... ]
        }, ...
      ],

      "external": { "plantid": {...} }   # if USE_PLANTID_HEALTH=1 and available
    }
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) IDENTIFY FIRST (for plant name & candidates) ----------
    identify_info = _run_identification(image_bgr, topk=CFG.topk)

    # ---------- 2) Plant.id-only mode ----------
    if CFG.mode == "plantid":
        result = _plantid_only(image_bgr)
        result.update(identify_info)
        return result

    # ---------- 3) Prepare crops (possibly YOLO) ----------
    crops = _leaf_crops_or_full(image_bgr)

    # ---------- 4) Detector-only mode ----------
    if CFG.mode == "detector":
        names = None
        try:
            yolo = load_yolo_health()
            names = getattr(yolo, "names", None)
        except Exception:
            names = None

        det_diseases = []
        for item in crops:
            det = item.get("detector") or {}
            cls_id = int(det.get("cls", -1))
            conf = float(det.get("conf", 0.0))
            raw_label = (
                names.get(cls_id) if isinstance(names, dict) and cls_id in names
                else names[cls_id] if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names)
                else str(cls_id)
            )
            nice = _pretty_disease_name(raw_label)
            det_diseases.append({"name": nice, "prob": conf})

        status = "healthy"
        top_conf = 0.0
        for d in det_diseases:
            if "healthy" not in d["name"].lower() and d["prob"] > top_conf:
                top_conf = d["prob"]
                status = "diseased"

        agg = {
            "status": status if det_diseases else "unknown",
            "confidence": float(top_conf),
            "diseases": sorted(det_diseases, key=lambda x: x["prob"], reverse=True)[:CFG.topk],
            "crops": [],
        }
        agg.update(identify_info)

        if CFG.use_plantid and plantid_health_assess is not None:
            try:
                ok, pid = plantid_health_assess(image_bgr)
                if ok and isinstance(pid, dict):
                    agg["external"] = pid
            except Exception:
                pass

        return agg

    # ---------- 5) Classifier / Hybrid mode ----------
    model, classes = load_classifier()

    # Graceful fallback if model couldn't be loaded
    if model is None:
        agg = {
            "status": "unknown",
            "confidence": 0.0,
            "diseases": [],
            "crops": [],
        }
        agg.update(identify_info)
        return agg
    responses: List[Dict[str, Any]] = []
    probs_all: List[float] = []

    for i, item in enumerate(crops):
        crop_bgr = item["crop"]
        crop_rgb = _to_rgb(crop_bgr)
        if is_onnx():
            t = None
            probs = model.predict(Image.fromarray(crop_rgb))
            idxs = [int(j) for j in np.argsort(-probs)[:CFG.topk]]
            vals = [float(probs[j]) for j in idxs]
        else:
            t = _preprocess_for_clf(crop_rgb, size=CFG.input_size)
            with torch.no_grad():
                logits = model(t)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                vals, idxs = _softmax_topk(logits, CFG.topk)

        # Build top-k pretty diseases
        top = []
        for p, idx in zip(vals, idxs):
            raw_name = classes[idx] if idx < len(classes) else str(idx)
            nice_name = _pretty_disease_name(raw_name)
            top.append({"name": nice_name, "prob": float(p)})

        pred_idx = int(idxs[0]) if idxs else 0
        raw_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        pretty_label = _pretty_disease_name(raw_label)
        pred_prob = float(vals[0]) if vals else 0.0
        probs_all.append(pred_prob)

        # CAM overlays (optional)
        overlay_paths: List[str] = []
        try:
            overlays = _generate_cam_overlays(model, crop_rgb, t, pred_idx)
        except Exception:
            overlays = []
        for name, img in overlays:
            save_p = os.path.join(out_dir, f"crop{i+1}_{name}.jpg")
            _ensure_dir(save_p)
            cv2.imwrite(save_p, img)
            overlay_paths.append(save_p.replace("\\", "/"))

        responses.append({
            "box": item["box"],
            "label": raw_label,
            "pretty_label": pretty_label,
            "prob": pred_prob,
            "topk": top,
            "xai": overlay_paths,
        })

    best = max(responses, key=lambda r: r["prob"]) if responses else None

    # Decide overall status
    if not responses:
        status = "unknown"
    elif best and "healthy" in str(best.get("label", "")).lower():
        status = "healthy"
    else:
        status = "diseased"

    agg: Dict[str, Any] = {
        "status": status,
        "confidence": float(np.mean(probs_all)) if probs_all else 0.0,
        # For UI & healthcare: best top-k diseases (pretty names)
        "diseases": (best["topk"] if best else []),
        "crops": responses,
    }

    # Attach identification info
    agg.update(identify_info)

    # Optionally merge Plant.id health
    if CFG.use_plantid and plantid_health_assess is not None:
        try:
            ok, pid = plantid_health_assess(image_bgr)
            if ok and isinstance(pid, dict):
                agg["external"] = pid
        except Exception:
            pass

    return agg
