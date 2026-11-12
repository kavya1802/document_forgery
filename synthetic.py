"""
Synthetic forged-document generator + evaluation (Accuracy, F1, IoU, Dice).

    python synthetic.py --n 10                 # synthetic benchmark
    python synthetic.py --images imgs/ --masks masks/   # CoMoFoD / DocTamper style
        (an image with no mask file of the same stem is treated as authentic)
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forgery_detector import Config, detect_forgery, load_image

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
WORDS = ("invoice total amount paid balance due account number student grade marks "
         "certificate issued date name signature approved department university").split()


def _font(i, size):
    try:
        return ImageFont.truetype(FONTS[i % len(FONTS)], size)
    except OSError:
        return ImageFont.load_default()


def make_document(rng, w=1200, h=1600, noise=2.0, font_i=0, quality=95):
    """Render a fake document page, then round-trip it through JPEG."""
    img = Image.new("RGB", (w, h), (246, 245, 240))
    d = ImageDraw.Draw(img)
    f = _font(font_i, 26)
    d.text((80, 70), "OFFICIAL CERTIFICATE / INVOICE", font=_font(font_i, 40), fill=(20, 20, 20))
    y = 170
    while y < h - 200:
        line = " ".join("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=rng.integers(3, 10)))
                        for _ in range(rng.integers(6, 11)))
        d.text((80, y), line, font=f, fill=(25, 25, 25))
        y += 46
    d.ellipse((820, 1300, 1020, 1500), outline=(30, 60, 160), width=6)       # stamp
    d.text((850, 1385), "APPROVED", font=_font(font_i, 28), fill=(30, 60, 160))
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, noise, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", arr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _save_q(img, q):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def forge(kind, rng):
    """Return (forged_image, ground_truth_mask)."""
    base = make_document(rng)
    H, W = base.shape[:2]
    gt = np.zeros((H, W), np.uint8)
    if kind == "copymove":      # duplicate a block of content (e.g. a stamp/figure) elsewhere
        patch = base[900:1120, 500:800].copy()
        x, y = 100, 400
        base[y:y + 220, x:x + 300] = patch
        gt[y:y + 220, x:x + 300] = 255
    elif kind == "splice":      # paste a region from a differently-made document
        other = _save_q(make_document(rng, noise=6.0, font_i=1, quality=60), 60)
        x, y, w, h = 300, 500, 520, 150
        base[y:y + h, x:x + w] = other[y:y + h, x:x + w]
        gt[y:y + h, x:x + w] = 255
    elif kind == "retype":      # erase a line and type over it in another font/noise
        x, y, w, h = 80, 600, 700, 60
        base[y:y + h, x:x + w] = (246, 245, 240)
        pil = Image.fromarray(base[:, :, ::-1])
        ImageDraw.Draw(pil).text((x, y + 8), "total amount paid 99999", font=_font(2, 30), fill=(0, 0, 0))
        base = np.asarray(pil)[:, :, ::-1].copy()
        gt[y:y + h, x:x + w] = 255
    return _save_q(base, 95), gt


def metrics(pred, gt):
    p, g = pred > 0, gt > 0
    inter, union = (p & g).sum(), (p | g).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    return iou, dice


def summarize(rows, name):
    """rows: list of (label, pred_label, iou, dice)."""
    y = np.array([r[0] for r in rows]); yp = np.array([r[1] for r in rows])
    tp = ((y == 1) & (yp == 1)).sum(); fp = ((y == 0) & (yp == 1)).sum(); fn = ((y == 1) & (yp == 0)).sum()
    acc = (y == yp).mean()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    forged = [r for r in rows if r[0] == 1]
    iou = np.mean([r[2] for r in forged]) if forged else float("nan")
    dice = np.mean([r[3] for r in forged]) if forged else float("nan")
    print(f"{name:<12} Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  IoU={iou:.3f}  Dice={dice:.3f}")


def evaluate(samples, cfg):
    rows = []
    for img, gt in samples:
        r = detect_forgery(img, None, cfg)
        pred_label = int(r.verdict != "AUTHENTIC")
        pred_mask = r.mask if pred_label else np.zeros_like(r.mask)
        if gt.shape != pred_mask.shape:
            gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        iou, dice = metrics(pred_mask, gt)
        rows.append((int(gt.any()), pred_label, iou, dice, r.score))
    return rows


def run_synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    cfg = Config()
    samples = []
    for _ in range(n):
        base = make_document(rng)
        samples.append((base, np.zeros(base.shape[:2], np.uint8)))      # authentic
        for kind in ("copymove", "splice", "retype"):
            samples.append(forge(kind, rng))
    rows = evaluate(samples, cfg)
    auth = [r[4] for r in rows if r[0] == 0]; forg = [r[4] for r in rows if r[0] == 1]
    print(f"mean score  authentic={np.mean(auth):.1f}   forged={np.mean(forg):.1f}")
    summarize(rows, "Synthetic")
    return rows


def run_folder(img_dir, mask_dir):
    cfg = Config()
    samples = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*"))):
        img = cv2.imread(p)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = glob.glob(os.path.join(mask_dir, stem + ".*"))
        gt = cv2.imread(mp[0], 0) if mp else np.zeros(img.shape[:2], np.uint8)
        samples.append((img, (gt > 127).astype(np.uint8) * 255))
    summarize(evaluate(samples, cfg), os.path.basename(os.path.normpath(img_dir)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="synthetic sets (4 images each)")
    ap.add_argument("--save", default=None, help="folder to dump a few sample forged images")
    ap.add_argument("--images"); ap.add_argument("--masks")
    a = ap.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        rng = np.random.default_rng(1)
        cv2.imwrite(os.path.join(a.save, "authentic.jpg"), make_document(rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        for k in ("copymove", "splice", "retype"):
            img, gt = forge(k, rng)
            cv2.imwrite(os.path.join(a.save, f"{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.save, f"{k}_mask.png"), gt)
    if a.images and a.masks:
        run_folder(a.images, a.masks)
    elif not a.save:
        run_synthetic(a.n)
