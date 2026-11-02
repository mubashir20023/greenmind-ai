"""Unit tests for the lite (ONNX) engine that need neither PyTorch nor model files:
pre-processing, CAM / Eigen-CAM and YOLO decoding (letterbox + NMS) with a fake ONNX session.
Run: python -m pytest tests/  (or python tests/test_engine.py)"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("GREENMIND_ENGINE", "onnx")
from app import engine as E  # noqa: E402


def test_preprocess_shapes_and_norm():
    x = E.preprocess(Image.new("RGB", (400, 300), (255, 0, 0)), 224, "crop", "imagenet")
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225], atol=1e-5)
    y = E.preprocess(Image.new("RGB", (300, 500)), 224, "square", "half")
    assert y.min() == -1.0 and y.max() == -1.0


def test_cam_and_eigen_cam_peak_at_hot_spot():
    c = object.__new__(E.OnnxClassifier)
    c.fc_w = np.random.rand(5, 8).astype(np.float32)
    f = np.random.rand(8, 7, 7).astype(np.float32)
    f[:, 2, 3] += 5
    for m in (c.cam(f, 1), E.OnnxClassifier.eigen_cam(f)):
        assert np.unravel_index(m.argmax(), m.shape) == (2, 3) and abs(m.max() - 1) < 1e-6


def test_yolo_letterbox_decoding_and_nms():
    y = object.__new__(E.OnnxYolo)
    y.names, y.imgsz, y.input_name = {0: "tree", 1: "leaf", 2: "flower"}, 640, "images"
    out = np.zeros((1, 7, 10), np.float32)
    out[0, :4, 0], out[0, 5, 0] = [320, 320, 100, 100], 0.9
    out[0, :4, 1], out[0, 5, 1] = [322, 321, 100, 100], 0.8  # duplicate, removed by NMS
    out[0, :4, 2], out[0, 6, 2] = [100, 320, 50, 50], 0.6

    class Sess:
        def run(self, _, feed):
            assert feed["images"].shape == (1, 3, 640, 640)
            return [out]
    y.sess = Sess()
    d = y.detect(Image.new("RGB", (1280, 640)), conf=0.25)
    assert [(c, n) for c, n, _, _ in d] == [(1, "leaf"), (2, "flower")]
    assert np.allclose(d[0][3], (540, 220, 740, 420))  # 2x scale, 160 px vertical padding removed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
