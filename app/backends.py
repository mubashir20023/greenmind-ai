from __future__ import annotations
import base64, io, json, os
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image

Alt = Dict[str, object]  # {label: str, score: float}


class Backend:
    name: str = "backend"
    timeout_sec: int = 25

    def available(self) -> bool:
        return True

    def identify(self, pil: Image.Image, topk: int = 3, organ_hint: Optional[str] = None) -> Tuple[Alt, List[Alt]]:
        raise NotImplementedError

    @staticmethod
    def _jpeg_bytes(pil: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


class PlantIdBackend(Backend):
    """Plan A: https://plant.id API (v3)"""
    name = "plant.id"

    def __init__(self):
        self.api_key = os.getenv("PLANT_ID_API_KEY")
        self.url = os.getenv("PLANT_ID_URL", "https://api.plant.id/v3/identification")

    def available(self) -> bool:
        return bool(self.api_key)

    def identify(self, pil: Image.Image, topk: int = 3, organ_hint: Optional[str] = None):
        img_b64 = base64.b64encode(self._jpeg_bytes(pil)).decode("ascii")
        body = {
            "images": [f"data:image/jpeg;base64,{img_b64}"],
            "similar_images": False,
        }
        if organ_hint:
            body["organs"] = [organ_hint]

        headers = {"Content-Type": "application/json", "Api-Key": self.api_key}
        r = requests.post(self.url, headers=headers, data=json.dumps(body), timeout=self.timeout_sec)
        r.raise_for_status()
        j = r.json()

        # Extract is_plant_probability if present (different shapes exist)
        def _is_plant_prob(payload):
            if isinstance(payload, dict):
                if "is_plant_probability" in payload:
                    return float(payload.get("is_plant_probability") or 0.0)
                if "result" in payload and isinstance(payload["result"], dict):
                    return float(payload["result"].get("is_plant_probability") or 0.0)
            return None

        is_plant_prob = _is_plant_prob(j)

        # Extract suggestions across known shapes
        suggestions = None
        if isinstance(j, dict):
            if "suggestions" in j:
                suggestions = j["suggestions"]
            elif "result" in j and isinstance(j["result"], dict):
                if "classification" in j["result"]:
                    suggestions = j["result"]["classification"].get("suggestions")
                else:
                    suggestions = j["result"].get("suggestions")
        if suggestions is None:
            suggestions = []

        items: List[Alt] = []
        for s in suggestions[: max(3, topk)]:
            name = (
                s.get("name")
                or (s.get("details") or {}).get("scientific_name")
                or (s.get("plant") or {}).get("scientific_name")
                or "Unknown"
            )
            prob = float(s.get("probability") or s.get("score") or 0.0)
            items.append({"label": name, "score": prob})

        if not items:
            items = [{"label": "Unknown", "score": 0.0}]

        best = items[0].copy()
        if is_plant_prob is not None:
            best["is_plant_prob"] = is_plant_prob

        return best, items[:topk]


class PlantNetBackend(Backend):
    """Plan B: Pl@ntNet v2 (all networks)"""
    name = "plantnet"

    def __init__(self):
        self.api_key = os.getenv("PLANTNET_API_KEY")
        self.endpoint = os.getenv("PLANTNET_URL", "https://my-api.plantnet.org/v2/identify/all")

    def available(self) -> bool:
        return bool(self.api_key)

    def identify(self, pil: Image.Image, topk: int = 3, organ_hint: Optional[str] = None):
        params = {"api-key": self.api_key}
        files = [("images", ("image.jpg", self._jpeg_bytes(pil), "image/jpeg"))]
        data = {}
        if organ_hint:
            data["organs"] = organ_hint  # Pl@ntNet accepts a single organ or repeated param

        r = requests.post(self.endpoint, params=params, files=files, data=data, timeout=self.timeout_sec)
        r.raise_for_status()
        j = r.json()

        # Typical shape: {"results": [{"score": 0.xx, "species": {...}}, ...]}
        results = []
        if isinstance(j, dict):
            results = j.get("results") or []

        items: List[Alt] = []
        for res in results[: max(3, topk)]:
            score = float(res.get("score") or 0.0)
            sp = res.get("species") or {}
            name = (
                sp.get("scientificNameWithoutAuthor")
                or sp.get("scientificName")
                or "Unknown"
            )
            items.append({"label": name, "score": score})

        if not items:
            items = [{"label": "Unknown", "score": 0.0}]

        return items[0], items[:topk]


def try_backends(pil: Image.Image, backends: List[Backend], topk: int = 3, organ_hint: Optional[str] = None) -> Tuple[str, Dict[str, object], List[Alt]]:
    """Try each backend in order; return (name, best, alts). If all fail, returns ("none", Unknown...)."""
    for be in backends:
        if not be.available():
            continue
        try:
            best, alts = be.identify(pil, topk=topk, organ_hint=organ_hint)
            if alts:
                return be.name, best, alts
        except Exception:
            continue
    return "none", {"label": "Unknown", "score": 0.0}, [{"label": "Unknown", "score": 0.0}]
