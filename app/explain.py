# app/explain.py
import io, math
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import timm
import cv2

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_model = None
_target_layers = None
_patch_hw = None  # (H, W) patch grid for ViT

def _get_model():
    """Load ViT-B/16 (ImageNet head ON) for CAM; cache model, target layer, and patch grid."""
    global _model, _target_layers, _patch_hw
    if _model is None:
        _model = timm.create_model("vit_base_patch16_224", pretrained=True)
        _model.eval().to(DEVICE)

        # Pick a reliable target layer from last block (name varies across timm versions)
        last_block = _model.blocks[-1]
        target_attr = None
        for cand in ("norm1", "ln1", "norm", "ln"):
            if hasattr(last_block, cand):
                target_attr = getattr(last_block, cand)
                break
        if target_attr is None:
            raise RuntimeError("Could not find a compatible target layer on ViT block")
        _target_layers = [target_attr]

        # Patch grid size, e.g., 14x14 for 224/16
        if hasattr(_model, "patch_embed") and hasattr(_model.patch_embed, "grid_size"):
            gs = _model.patch_embed.grid_size
            _patch_hw = (int(gs[0]), int(gs[1]))
        else:
            _patch_hw = (14, 14)

    return _model, _target_layers, _patch_hw

def _vit_reshape_transform(tokens: torch.Tensor, hw=None):
    """ViT tokens [B,N,C] -> [B,C,H,W], dropping CLS token."""
    if tokens.ndim == 4:
        return tokens  # already [B,C,H,W]
    b, n, c = tokens.shape
    tokens = tokens[:, 1:, :]  # drop CLS
    if hw is None:
        s = int(math.sqrt(tokens.shape[1]))
        h = w = s
    else:
        h, w = hw
    tokens = tokens.permute(0, 2, 1).reshape(b, c, h, w)
    return tokens

def generate_cam_overlay(pil_image: Image.Image, method: str = "gradcam") -> bytes:
    """
    Return a transparent PNG (RGBA) heatmap:
      - RGB = colored CAM
      - A (alpha) = CAM intensity
    Client blends it with the slider.
    method: 'gradcam' (class-specific) or 'eigen' (class-agnostic)
    """
    model, target_layers, patch_hw = _get_model()

    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    # Choose CAM implementation
    CamImpl = GradCAM if method == "gradcam" else EigenCAM

    # Build CAM (no use_cuda; device is inferred)
    with CamImpl(
        model=model,
        target_layers=target_layers,
        reshape_transform=lambda t: _vit_reshape_transform(t, patch_hw),
    ) as cam:

        targets = None
        if method == "gradcam":
            with torch.inference_mode():
                logits = model(x)
                pred_id = int(torch.argmax(logits, dim=1).item())
            targets = [ClassifierOutputTarget(pred_id)]

        cam_map = cam(input_tensor=x, targets=targets)  # [1, H, W]
        cam_map = cam_map[0]

    # ---- create RGBA heatmap (not pre-blended) ----
    cam_norm = cam_map - cam_map.min()
    if cam_norm.max() > 0:
        cam_norm = cam_norm / cam_norm.max()

    cam_uint8 = (cam_norm * 255).astype(np.uint8)                # (H, W)
    heat_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)    # (H, W, 3) BGR
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)         # (H, W, 3) RGB

    # Alpha = CAM intensity
    alpha = cam_uint8                                            # (H, W)
    heat_rgba = np.dstack([heat_rgb, alpha])                     # (H, W, 4) RGBA

    # Resize to client display size
    heat_rgba = cv2.resize(heat_rgba, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    out = Image.fromarray(heat_rgba, mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
