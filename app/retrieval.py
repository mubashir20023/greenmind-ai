from __future__ import annotations
import os, io, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

# All optional: if not installed or files missing, we degrade to "unavailable"
try:
    import faiss                   # pip install faiss-cpu
    import open_clip               # pip install open-clip-torch
except Exception:                   # pragma: no cover
    faiss = None
    open_clip = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PATH = Path(os.getenv("RETR_INDEX_PATH", "models/retrieval/faiss.index"))
META_PATH  = Path(os.getenv("RETR_META_PATH",  "models/retrieval/gallery_meta.json"))
MODEL_NAME = os.getenv("RETR_MODEL_NAME", "ViT-H-14")
PRETRAINED = os.getenv("RETR_PRETRAINED", "laion2b_s32b_b79k")

class ClipRetrieval:
    def __init__(self):
        self.ok = bool(faiss and open_clip and INDEX_PATH.exists() and META_PATH.exists())
        self._why = None
        if not self.ok:
            miss = []
            if faiss is None: miss.append("faiss")
            if open_clip is None: miss.append("open_clip_torch")
            if not INDEX_PATH.exists(): miss.append(str(INDEX_PATH))
            if not META_PATH.exists():  miss.append(str(META_PATH))
            self._why = "missing: " + ", ".join(miss)
            self.model = None
            self.index = None
            self.meta  = []
            return

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
        )
        self.model.eval()
        self.index = faiss.read_index(str(INDEX_PATH))
        with META_PATH.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)  # list of {"species": "..."} aligned with index IDs

    def available(self) -> bool:
        return self.ok

    def reason_unavailable(self) -> str:
        return self._why or ""

    @torch.inference_mode()
    def _embed(self, pil) -> torch.Tensor:
        x = self.preprocess(pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        z = self.model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        return z

    @torch.inference_mode()
    def scores(self, pil, k: int = 64) -> Dict[str, float]:
        """Return softmax scores over species using max-hit per species from top-k neighbors."""
        if not self.available():
            return {}
        z = self._embed(pil).cpu().numpy().astype("float32")
        D, I = self.index.search(z, k)           # higher is better if index is IP
        I, D = I[0], D[0]
        by_sp = {}
        for idx, sim in zip(I, D):
            if idx < 0: continue
            sp = self.meta[idx]["species"]
            by_sp[sp] = max(by_sp.get(sp, -1e9), float(sim))
        if not by_sp: return {}
        sims = torch.tensor(list(by_sp.values()))
        probs = torch.softmax(sims, dim=0).tolist()
        return dict(zip(by_sp.keys(), probs))

RETR = ClipRetrieval()
