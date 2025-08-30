# app/detector.py
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional
from PIL import Image

# -------- Config (env-overridable) --------
YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", "runs/detect/train/weights/best.pt"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_MAX_DETS = int(os.getenv("YOLO_MAX_DETS", "100"))
YOLO_MAX_CROPS = int(os.getenv("YOLO_MAX_CROPS", "5"))

# Only keep plant-relevant classes; your dataset uses {tree, leaf, flower}
ALLOWED_CLASSES = {c.strip().lower() for c in os.getenv(
    "YOLO_ALLOWED", "tree,leaf,flower"
).split(",")}

class _Detector:
    """
    Thin wrapper around Ultralytics YOLO.
    Exposes:
      - is_available() -> bool
      - detect(img: PIL.Image) -> List[Dict] with keys: cls, conf, box(x1,y1,x2,y2), crop(PIL)
    """
    def __init__(self):
        self._weights = YOLO_WEIGHTS
        self._why_unavailable: Optional[str] = None
        self._available = self._weights.exists()
        if not self._available:
            self._why_unavailable = f"weights not found at {self._weights}"

    def is_available(self) -> bool:
        return self._available

    def reason_unavailable(self) -> str:
        return self._why_unavailable or ""

    @lru_cache(maxsize=1)
    def _yolo(self):
        if not self.is_available():
            return None
        try:
            from ultralytics import YOLO  # lazy import
            return YOLO(str(self._weights))
        except Exception as e:
            self._available = False
            self._why_unavailable = f"failed to load YOLO: {e}"
            return None

    @staticmethod
    def _clip_box(xyxy, W, H):
        x1, y1, x2, y2 = map(float, xyxy)
        x1 = max(0.0, min(x1, W - 1))
        y1 = max(0.0, min(y1, H - 1))
        x2 = max(0.0, min(x2, W - 1))
        y2 = max(0.0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)

    def detect(self, img: Image.Image) -> List[Dict]:
        """
        Return list of dicts: {'cls': str, 'conf': float, 'box': (x1,y1,x2,y2), 'crop': PIL.Image}
        Only for classes in ALLOWED_CLASSES. May return [].
        """
        if not self.is_available():
            return []

        yolo = self._yolo()
        if yolo is None:
            return []

        # Run prediction in-memory on PIL Image
        res = yolo.predict(
            source=img,
            conf=YOLO_CONF,
            max_det=YOLO_MAX_DETS,
            verbose=False,
        )
        if not res:
            return []

        r = res[0]
        names = r.names
        W, H = img.size

        out: List[Dict] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        for b in boxes:
            cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            cls_name = str(names.get(cls_id, cls_id)).lower()
            if cls_name not in ALLOWED_CLASSES:
                continue

            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy, "tolist") else list(b.xyxy[0])
            box = self._clip_box(xyxy, W, H)
            if not box:
                continue

            x1, y1, x2, y2 = box
            crop = img.crop((x1, y1, x2, y2))
            out.append({"cls": cls_name, "conf": conf, "box": box, "crop": crop})

            if len(out) >= YOLO_MAX_CROPS:
                break
        return out

DETECTOR = _Detector()
