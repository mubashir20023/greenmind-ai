# app/backends.py
from __future__ import annotations

import os
import io
import json
import base64
from typing import List, Dict, Optional, Tuple

import cv2
import requests
from PIL import Image

Alt = Dict[str, object]  # {label: str, score: float}


# ---------------- Shared helpers ----------------
def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _norm_label(s: Optional[str]) -> str:
    if not s:
        return "Unknown"
    return " ".join(s.strip().split())


def _pick_organ_hint(hint: Optional[str]) -> Optional[str]:
    if not hint:
        return None
    hint = hint.strip().lower()
    if hint in ("leaf", "flower", "fruit"):
        return hint
    return None


# ---------------- Base backend ----------------
class Backend:
    name: str = "backend"

    def available(self) -> bool:
        return True

    def identify(
        self,
        pil: Image.Image,
        topk: int = 3,
        organ_hint: Optional[str] = None
    ) -> Tuple[Alt, List[Alt]]:
        raise NotImplementedError


# ===================== Plant.id (v3) =====================
class PlantIdBackend(Backend):
    name = "plant.id"

    def __init__(self):
        self.api_key  = (os.getenv("PLANT_ID_API_KEY") or os.getenv("PLANTID_API_KEY") or "").strip()
        self.url      = (os.getenv("PLANT_ID_URL") or "https://api.plant.id/v3/identification").strip()
        self.timeout  = float(os.getenv("PLANTID_TIMEOUT", "15"))

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.url)

    def identify(
        self,
        pil: Image.Image,
        topk: int = 3,
        organ_hint: Optional[str] = None
    ) -> Tuple[Alt, List[Alt]]:
        if not self.available():
            raise RuntimeError("PlantIdBackend not available")

        img_b64 = base64.b64encode(_pil_to_jpeg_bytes(pil)).decode("ascii")
        payload = {
            "images": [f"data:image/jpeg;base64,{img_b64}"],
            "similar_images": False,
            "classification_level": "species",
            "health": False,
            "details": ["common_names", "name_authority", "taxonomy"],
        }

        organ = _pick_organ_hint(organ_hint)
        if organ:
            payload["organs"] = [organ]

        headers = {"Api-Key": self.api_key, "User-Agent": "greenmind-ai/1.0"}

        r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        isp = 0.0
        if isinstance(data, dict):
            if "is_plant_probability" in data:
                isp = float(data.get("is_plant_probability") or 0.0)
            elif isinstance(data.get("result"), dict):
                isp = float(data["result"].get("is_plant_probability") or 0.0)

        suggestions = []
        if isinstance(data, dict):
            if isinstance(data.get("suggestions"), list):
                suggestions = data["suggestions"]
            else:
                res = data.get("result")
                if isinstance(res, dict):
                    cls = res.get("classification")
                    if isinstance(cls, dict) and isinstance(cls.get("suggestions"), list):
                        suggestions = cls["suggestions"]
                    elif isinstance(res.get("suggestions"), list):
                        suggestions = res["suggestions"]

        items: List[Alt] = []
        for s in (suggestions or [])[: max(1, topk)]:
            plant_details = s.get("plant_details") or {}
            name = (
                plant_details.get("scientific_name")
                or s.get("name")
                or (s.get("plant") or {}).get("scientific_name")
                or "Unknown"
            )
            prob = float(s.get("probability") or s.get("score") or 0.0)
            items.append({"label": _norm_label(name), "score": prob})

        if not items:
            items = [{"label": "Unknown", "score": 0.0}]

        best = dict(items[0])
        best["is_plant_prob"] = isp
        return best, items[: topk]


# ===================== Pl@ntNet (v2 /identify/all) =====================
class PlantNetBackend(Backend):
    name = "plantnet"

    def __init__(self):
        self.api_key  = (os.getenv("PLANTNET_API_KEY") or "").strip()
        self.url      = (os.getenv("PLANTNET_URL") or "https://my-api.plantnet.org/v2/identify/all").strip()
        self.timeout  = float(os.getenv("PLANTNET_TIMEOUT", "15"))

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.url)

    def identify(
        self,
        pil: Image.Image,
        topk: int = 3,
        organ_hint: Optional[str] = None
    ) -> Tuple[Alt, List[Alt]]:
        if not self.available():
            raise RuntimeError("PlantNetBackend not available")

        organ = _pick_organ_hint(organ_hint)
        params = {"api-key": self.api_key}
        files = [("images", ("query.jpg", _pil_to_jpeg_bytes(pil), "image/jpeg"))]
        data = [("organs", organ)] if organ else []

        r = requests.post(self.url, params=params, files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        results = data.get("results") or [] if isinstance(data, dict) else []
        items: List[Alt] = []
        for s in results[: max(1, topk)]:
            score = float(s.get("score") or 0.0)
            sp = s.get("species") or {}
            name = (
                sp.get("scientificNameWithoutAuthor")
                or sp.get("scientificName")
                or (sp.get("genus", {}).get("scientificName") if isinstance(sp.get("genus"), dict) else None)
                or "Unknown"
            )
            items.append({"label": _norm_label(name), "score": score})

        if not items:
            items = [{"label": "Unknown", "score": 0.0}]

        best = dict(items[0])
        return best, items[: topk]


# ===================== First-success helper =====================
def try_backends(
    pil: Image.Image,
    backends: List[Backend],
    topk: int = 3,
    organ_hint: Optional[str] = None
) -> Tuple[str, Dict[str, object], List[Alt]]:
    for be in backends:
        try:
            if be.available():
                best, alts = be.identify(pil, topk=topk, organ_hint=organ_hint)
                if alts:
                    return be.name, best, alts
        except Exception:
            continue
    return "none", {"label": "Unknown", "score": 0.0}, [{"label": "Unknown", "score": 0.0}]


# ===================== Health Assessment =====================
def health_assess(image_bgr):
    api_key = (os.getenv("PLANT_ID_API_KEY") or os.getenv("PLANTID_API_KEY") or "").strip()
    url     = (os.getenv("PLANT_ID_URL") or "https://api.plant.id/v3/identification").strip()
    timeout = float(os.getenv("PLANTID_TIMEOUT", "15"))

    if not api_key or not url:
        return False, {}

    ok, enc = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return False, {}
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")

    payload = {
        "images": [f"data:image/jpeg;base64,{b64}"],
        "classification_level": "species",
        "similar_images": False,
        "health": True,
        "modifiers": ["health"]
    }

    headers = {"Api-Key": api_key, "Content-Type": "application/json", "User-Agent": "greenmind-ai/1.0"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not resp.ok:
            return False, {}

        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

        ha = {}
        if isinstance(data, dict):
            ha = data.get("health_assessment") or {}
            if not ha:
                res = data.get("result")
                if isinstance(res, dict):
                    ha = res.get("health_assessment") or {}

        is_healthy = ha.get("is_healthy", None)

        diseases_raw = ha.get("diseases") or []
        diseases_norm = []
        for d in diseases_raw:
            if isinstance(d, dict):
                name = d.get("name") or d.get("id") or "unknown"
                prob = d.get("prob") or d.get("probability") or 0.0
                try:
                    prob = float(prob)
                except Exception:
                    prob = 0.0
                diseases_norm.append({"name": str(name), "prob": prob})
            else:
                diseases_norm.append({"name": str(d), "prob": 0.0})

        return True, {"plantid": {"is_healthy": is_healthy, "diseases": diseases_norm}}

    except Exception:
        return False, {}
