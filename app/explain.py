# app/explain.py
import io, math
import numpy as np
from PIL import Image, ImageOps
import cv2

from app.engine import is_onnx
from app import engine as onnx_engine

IMG_SIZE = 224

if is_onnx():  # lite engine: CAM / Eigen-CAM computed with NumPy from the ONNX model outputs
    torch = transforms = timm = GradCAM = EigenCAM = ClassifierOutputTarget = None
    DEVICE = None
else:
    import torch
    from torchvision import transforms
    import timm
    from pytorch_grad_cam import GradCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = None if is_onnx() else transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_model = None
_target_layers = None
_patch_hw = None  # (H, W) patch grid for ViT fallback
_is_vit = False

def _get_model():
    """
    Explain the SAME model that makes the local species prediction (app.identify), using its last
    convolutional layer as the Grad-CAM / Eigen-CAM target. Only if no local classifier is available
    do we fall back to a generic ImageNet ViT-B/16.
    """
    global _model, _target_layers, _patch_hw, _is_vit
    if _model is not None:
        return _model, _target_layers, _patch_hw

    local = None
    try:
        from app.identify import _load_model
        local = _load_model()
    except Exception as e:  # pragma: no cover
        print(f"[explain] local classifier unavailable: {e}")

    if local is not None:
        convs = [m for m in local.modules() if isinstance(m, torch.nn.Conv2d)]
        if convs:
            _model, _target_layers, _patch_hw, _is_vit = local, [convs[-1]], None, False
            return _model, _target_layers, _patch_hw

    # Fallback: generic ImageNet ViT (explanations are then NOT of the plant classifier)
    _model = timm.create_model("vit_base_patch16_224", pretrained=True)
    _model.eval().to(DEVICE)
    last_block = _model.blocks[-1]
    target_attr = None
    for cand in ("norm1", "ln1", "norm", "ln"):
        if hasattr(last_block, cand):
            target_attr = getattr(last_block, cand)
            break
    if target_attr is None:
        raise RuntimeError("Could not find a compatible target layer on ViT block")
    _target_layers = [target_attr]
    if hasattr(_model, "patch_embed") and hasattr(_model.patch_embed, "grid_size"):
        gs = _model.patch_embed.grid_size
        _patch_hw = (int(gs[0]), int(gs[1]))
    else:
        _patch_hw = (14, 14)
    _is_vit = True
    return _model, _target_layers, _patch_hw

def _vit_reshape_transform(tokens, hw=None):
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
    With the lite (ONNX) engine, 'gradcam' is served by CAM, its gradient-free equivalent for this architecture.
    """
    if is_onnx():
        sm = onnx_engine.species_model()
        if sm is None:
            raise RuntimeError("ONNX species model not found")
        _, _, cam_map = sm.predict_with_cam(pil_image, method="eigen" if method == "eigen" else "cam")
        return _heat_png(cam_map)

    model, target_layers, patch_hw = _get_model()

    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    # Choose CAM implementation
    CamImpl = GradCAM if method == "gradcam" else EigenCAM

    # Build CAM (no use_cuda; device is inferred)
    with CamImpl(
        model=model,
        target_layers=target_layers,
        reshape_transform=(lambda t: _vit_reshape_transform(t, patch_hw)) if _is_vit else None,
    ) as cam:

        targets = None
        if method == "gradcam":
            with torch.inference_mode():
                logits = model(x)
                pred_id = int(torch.argmax(logits, dim=1).item())
            targets = [ClassifierOutputTarget(pred_id)]

        cam_map = cam(input_tensor=x, targets=targets)  # [1, H, W]
        cam_map = cam_map[0]

    return _heat_png(cam_map)


def _heat_png(cam_map) -> bytes:
    """Transparent RGBA PNG (colour = CAM, alpha = intensity), IMG_SIZE x IMG_SIZE, aligned with the
    centre crop the classifier sees. The client blends it over the photo."""
    cam_map = np.asarray(cam_map, dtype=np.float32)
    if cam_map.shape != (IMG_SIZE, IMG_SIZE):  # e.g. 7x7 CAM from the ONNX engine: smooth upsampling first
        cam_map = cv2.resize(cam_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
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
