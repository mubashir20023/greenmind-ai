#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an image classifier from a manifest.csv (made by tools/make_split.py).
One script for both GreenMind models:

  Species (ResNet-18, ImageNet normalisation -> compatible with app/identify.py)
    python tools/train_classifier.py --manifest DATA/manifest.csv --arch resnet18 ^
        --norm imagenet --epochs 25 --out runs/species_resnet18

  Disease (EfficientNet-B0, [-1,1] normalisation, full-model save -> app/health.py)
    python tools/train_classifier.py --manifest DATA/manifest.csv --arch efficientnet_b0 ^
        --norm half --resize_mode square --epochs 20 --save_full --out runs/disease_effb0

Everything needed for the paper is written to --out:
  best.pth (state_dict), model_full.pt (if --save_full), classes.json,
  idx_to_species.json, config.json, history.csv, and test metrics via evaluate.py.
Training resumes automatically from --out/last.pth if it exists.
"""

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm_data import ManifestDataset, build_transforms, pick_device, read_manifest  # noqa: E402

import timm  # noqa: E402


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate_loader(model, loader, device, amp):
    model.eval()
    correct1 = correct5 = n = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            loss = crit(out.float(), y)
        loss_sum += loss.item() * y.size(0)
        k = min(5, out.size(1))
        top = out.topk(k, 1).indices
        correct1 += (top[:, 0] == y).sum().item()
        correct5 += (top == y[:, None]).any(1).sum().item()
        n += y.size(0)
    return loss_sum / max(1, n), correct1 / max(1, n), correct5 / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="Base folder for relative paths (default: manifest folder)")
    ap.add_argument("--arch", default="resnet18", help="Any timm model name")
    ap.add_argument("--norm", choices=["imagenet", "half"], default="imagenet")
    ap.add_argument("--resize_mode", choices=["crop", "square"], default="crop")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6, help="Early stop after N epochs without val improvement")
    ap.add_argument("--workers", type=int, default=4, help="Use 0 on Windows if loading hangs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="Debug only: random init")
    ap.add_argument("--save_full", action="store_true", help="Also save the whole model (needed by app/health.py)")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="Debug only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    amp = (device.type == "cuda") and not args.no_amp

    train_rows = read_manifest(args.manifest, "train", args.root)
    val_rows = read_manifest(args.manifest, "val", args.root)
    classes = sorted({l for _, l in train_rows})
    if not val_rows:
        sys.exit("Manifest has no 'val' rows; val is needed for checkpoint selection.")

    train_ds = ManifestDataset(train_rows, classes, build_transforms(args.img_size, args.norm, True))
    val_ds = ManifestDataset(val_rows, classes,
                             build_transforms(args.img_size, args.norm, False, args.resize_mode))
    print(f"[data] classes={len(classes)} train={len(train_ds)} val={len(val_ds)} "
          f"(val rows with unseen class: {val_ds.n_unknown}) device={device} amp={amp}")

    (out / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "idx_to_species.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(classes)}, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg = vars(args).copy()
    cfg.update(num_classes=len(classes), n_train=len(train_ds), n_val=len(val_ds),
               torch=torch.__version__, timm=timm.__version__,
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=len(train_ds) > args.batch_size,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers,
                        pin_memory=pin, persistent_workers=args.workers > 0)

    model = timm.create_model(args.arch, pretrained=not args.no_pretrained, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = len(train_ld) if not args.max_steps_per_epoch else min(len(train_ld), args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_epochs * steps_per_epoch)

    start_epoch, best, bad, step = 1, -1.0, 0, 0
    hist_path = out / "history.csv"
    last = out / "last.pth"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if amp and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best, bad, step = st["epoch"] + 1, st["best"], st["bad"], st["step"]
        print(f"[resume] from epoch {st['epoch']} (best val acc {best:.4f})")
    elif hist_path.exists():
        hist_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tl = tc = tn = 0
        for i, (x, y) in enumerate(train_ld):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, args.lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outp = model(x)
                loss = crit(outp.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            tl += loss.item() * y.size(0)
            tc += (outp.argmax(1) == y).sum().item()
            tn += y.size(0)
            if i % 200 == 0:
                print(f"  ep{epoch} {i}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)

        vl, v1, v5 = evaluate_loader(model, val_ld, device, amp)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] train_loss {tl/tn:.4f} train_acc {tc/tn:.4f} | "
              f"val_loss {vl:.4f} val_top1 {v1:.4f} val_top5 {v5:.4f} | {dt/60:.1f} min", flush=True)
        new = not hist_path.exists()
        with hist_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5", "lr", "minutes"])
            w.writerow([epoch, f"{tl/tn:.5f}", f"{tc/tn:.5f}", f"{vl:.5f}", f"{v1:.5f}", f"{v5:.5f}",
                        f"{opt.param_groups[0]['lr']:.2e}", f"{dt/60:.2f}"])

        if v1 > best:
            best, bad = v1, 0
            torch.save(model.state_dict(), out / "best.pth")
            print(f"  [save] best.pth (val_top1={best:.4f})")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad, "step": step}, last)
        if bad >= args.patience:
            print(f"[early stop] no val improvement for {args.patience} epochs")
            break

    # final: reload best, optional full save, evaluate on test split
    model.load_state_dict(torch.load(out / "best.pth", map_location=device, weights_only=True))
    if args.save_full:
        model.eval().cpu()
        torch.save(model, out / "model_full.pt")
        print(f"[save] {out / 'model_full.pt'} (drop-in for app/health.py)")
    print(f"[done] best val top-1 = {best:.4f}")

    # free GPU memory before the evaluation subprocess starts
    del model, opt, scaler, train_ld, val_ld
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if read_manifest(args.manifest, "test", args.root):
        cmd = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
               "--ckpt", str(out / "best.pth"), "--arch", args.arch, "--classes", str(out / "classes.json"),
               "--manifest", args.manifest, "--split", "test", "--norm", args.norm,
               "--resize_mode", args.resize_mode, "--img_size", str(args.img_size),
               "--workers", str(args.workers), "--out", str(out / "eval_test")]
        if args.root:
            cmd += ["--root", args.root]
        print("[eval]", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
