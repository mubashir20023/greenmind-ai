from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WEIGHTS = Path(os.getenv("PLANTNESS_WEIGHTS", "models/plantness_mnv3.pth"))
_IMG_SIZE = int(os.getenv("PLANTNESS_IMG_SIZE", "224"))

_T = transforms.Compose([
    transforms.Resize(_IMG_SIZE),
    transforms.CenterCrop(_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class Plantness:
    def __init__(self):
        self._why = None
        self.ok = _WEIGHTS.exists()
        self.model = None
        if not self.ok:
            self._why = f"weights not found at {_WEIGHTS}"
            return
        try:
            self.model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
            state = torch.load(_WEIGHTS, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            self.model.eval().to(_DEVICE)
        except Exception as e:
            self.ok = False
            self._why = f"failed to load: {e}"

    def available(self) -> bool:
        return self.ok

    def reason_unavailable(self) -> str:
        return self._why or ""

    @torch.inference_mode()
    def score(self, pil: Image.Image) -> float:
        """Returns P(plant) in [0,1]."""
        if not self.available():
            return 1.0  # neutral if not available
        x = _T(pil.convert("RGB")).unsqueeze(0).to(_DEVICE)
        logits = self.model(x)          # [1,2] ; assume index 1 == plant
        p = F.softmax(logits, dim=1)[0][1].item()
        return float(p)

PLANTNESS = Plantness()
