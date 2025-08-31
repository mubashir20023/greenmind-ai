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

        # Try multiple known shapes
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
        return items[0], items[:topk]


class PlantNetBackend(Backend):
    """Plan B: Pl@ntNet v2 (all networks)"""
    name = "plantnet"

    def __init__(self):
        self.api_key = os.getenv("PLANTNET_API_KEY")
        self.endpoint = os.getenv("PLANTNET_URL", "https://my-api.plantnet.org/v2/identify/all")

    def available(self) -> bool:
        return bool(self.api_key)

    def identify(self, pil: Image.Image, topk: int = 3, organ_hint: Optional[str] = None):
        img_bytes = self._jpeg_bytes(pil)
        files = [("images", ("image.jpg", img_bytes, "image/jpeg"))]
        data = {"organs": organ_hint or "auto"}
        params = {"api-key": self.api_key}
        r = requests.post(self.endpoint, params=params, data=data, files=files, timeout=self.timeout_sec)
        r.raise_for_status()
        j = r.json()

        items: List[Alt] = []
        for s in (j.get("results") or [])[: max(3, topk)]:
            sp = s.get("species") or {}
            name = (
                sp.get("scientificNameWithoutAuthor")
                or sp.get("scientificName")
                or sp.get("genus")
                or "Unknown"
            )
            score = float(s.get("score") or 0.0)
            items.append({"label": name.strip(), "score": score})
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
            # swallow and continue to next backend
            continue
    return "none", {"label": "Unknown", "score": 0.0}, [{"label": "Unknown", "score": 0.0}]
