#!/usr/bin/env python
import os, re, shutil
from pathlib import Path
from PIL import Image

# ----- EDIT ME: map folder name keywords to detector class id -----
# 0 tree, 1 leaf, 2 flower
KEYWORD_TO_CLASS = {
    r"tree": 0,
    r"\bleaf\b": 1,
    r"rose|sunflower|flower|blossom|bloom": 2,
}

SRC_DIRS = [
    Path("data/gallery"),          # your small demo gallery
    Path("data/plants/train"),     # species folders (optional)
    Path("data/plants/val"),       # species folders (optional)
]

OUT_ROOT = Path("data/plant_parts")
for split in ["train", "val"]:
    (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

def pick_class_from_name(name: str):
    n = name.lower()
    for pat, cid in KEYWORD_TO_CLASS.items():
        if re.search(pat, n):
            return cid
    return None  # unknown/skip

def yolo_full_box_label(w, h, cls_id, margin=0.05):
    # central 90% box by default (margin=0.05 on each side)
    x1 = margin * w
    y1 = margin * h
    x2 = (1 - margin) * w
    y2 = (1 - margin) * h
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

def process_image(p: Path, split: str, cls_id: int):
    out_img = OUT_ROOT / split / "images" / p.name
    out_lbl = OUT_ROOT / split / "labels" / (p.stem + ".txt")
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        # copy image
        if not out_img.exists():
            shutil.copy2(p, out_img)
        # write label
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write(yolo_full_box_label(w, h, cls_id))
        return True
    except Exception as e:
        print("[skip]", p, e)
        return False

def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for r, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                yield Path(r) / fn

def main():
    import random
    random.seed(42)
    all_imgs = []
    for src in SRC_DIRS:
        if src.exists():
            for cls_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
                cls_id = pick_class_from_name(cls_dir.name)
                if cls_id is None:
                    continue
                for imgp in iter_images(cls_dir):
                    all_imgs.append((imgp, cls_id))
    random.shuffle(all_imgs)

    # 80/20 split
    n_train = int(0.8 * len(all_imgs))
    train_set = all_imgs[:n_train]
    val_set   = all_imgs[n_train:]

    ok = 0
    for imgp, cid in train_set:
        ok += process_image(imgp, "train", cid)
    for imgp, cid in val_set:
        ok += process_image(imgp, "val", cid)

    print(f"[done] Wrote weak labels for {ok} images into {OUT_ROOT}")

if __name__ == "__main__":
    main()
