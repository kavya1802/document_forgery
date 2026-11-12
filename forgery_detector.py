"""
Document Forgery Detection & Localization using multi-domain image analysis.

Pipeline (matches the slides):
  Input -> Preprocess -> 6 analyzers (ELA, copy-move, noise, texture, FFT/DCT,
  splicing) -> robust normalise + weighted fusion -> Otsu localisation -> report.

Classical DIP only: OpenCV + NumPy + scikit-image + SciPy. CPU only, no training.

Usage:
    python forgery_detector.py path/to/document.jpg --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.fft import dctn
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    max_side: int = 1600          # I1 = resize(I0, 1600)
    ela_quality: int = 90         # ELA(I0, q = 90)
    block: int = 32               # block size for block-wise cues (multiple of 8)
    blank_std: float = 4.0        # blocks with std below this count as "blank paper"
    weights: dict = field(default_factory=lambda: {
        "copymove": 0.25, "ela": 0.20, "noise": 0.15,
        "freq": 0.15, "splicing": 0.15, "texture": 0.10,
    })                            # must sum to 1 (slide 7)
    # Each cue's map is stretched with (M - P1) / max(P99 - P1, floor).
    # The floor stops a clean page from being stretched into "suspicious".
    norm_floor: dict = field(default_factory=lambda: {
        "copymove": 1.0, "ela": 6.0, "noise": 6.0,
        "freq": 6.0, "splicing": 6.0, "texture": 6.0,
    })
    top_frac: float = 0.02        # score = 100 * mean(top 2 % of S)
    score_gain: float = 1.6       # calibrated on the synthetic set (slide 8: "weight tuning")
    thr_review: float = 30.0      # 0-30 AUTHENTIC, 30-60 review, 60-100 forged
    thr_forged: float = 60.0
    min_tau: float = 0.15         # config floor for the Otsu threshold
    min_area_frac: float = 0.0005 # drop boxes smaller than this fraction of the page
    # copy-move
    cm_ratio: float = 0.75
    cm_min_shift: float = 40.0
    cm_min_inliers: int = 15


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_image(path: str, cfg: Config):
    I0 = cv2.imread(path, cv2.IMREAD_COLOR)
    if I0 is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    H, W = I0.shape[:2]
    s = cfg.max_side / max(H, W)
    I1 = cv2.resize(I0, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else I0.copy()
    return I0, I1


def pad_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    H, W = img.shape[:2]
    ph, pw = (-H) % m, (-W) % m
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT) if (ph or pw) else img


def block_view(a: np.ndarray, B: int) -> np.ndarray:
    """(H, W) -> (h, w, B, B) non-overlapping blocks. H, W must be multiples of B."""
    h, w = a.shape[0] // B, a.shape[1] // B
    return a.reshape(h, B, w, B).transpose(0, 2, 1, 3)


def robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    """|x - median| / (1.4826 * MAD), with a floor on the scale."""
    med = np.median(x)
    s = max(1.4826 * np.median(np.abs(x - med)), floor)
    return np.abs(x - med) / s


def class_z(x: np.ndarray, cls: np.ndarray, floor: float, min_n: int = 15) -> np.ndarray:
    """Robust z-score computed separately for blank-paper and content blocks,
    so text is only compared with text and background with background."""
    z = np.zeros(x.shape, np.float32)
    for c in (0, 1):
        m = cls == c
        if m.sum() >= min_n:
            z[m] = robust_z(x[m], floor)
    return z


def robust_normalize(M: np.ndarray, floor: float) -> np.ndarray:
    p1, p99 = np.percentile(M, (1, 99))
    return np.clip((M - p1) / max(p99 - p1, floor), 0, 1).astype(np.float32)


def grid_to_full(m: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
    """Upsample a block-grid map to pixel resolution and crop the padding."""
    h, w = m.shape
    up = cv2.resize(m.astype(np.float32), (w * B, h * B), interpolation=cv2.INTER_LINEAR)
    return up[:H, :W]


# --------------------------------------------------------------------------- #
# Shared block features (used by noise, texture, splicing)
# --------------------------------------------------------------------------- #
def block_features(Gp: np.ndarray, B: int, blank_std: float) -> dict:
    g = Gp.astype(np.float32)
    blocks = block_view(g, B)
    h, w = blocks.shape[:2]
    F = {"h": h, "w": w}
    F["mean"] = blocks.mean((2, 3))
    F["std"] = blocks.std((2, 3))
    F["cls"] = (F["std"] > blank_std).astype(np.uint8)   # 0 = blank paper, 1 = content

    # Noise: sigma_b = 1.4826 * MAD(I - med3(I)), measured on flat pixels only
    # (text edges would otherwise dominate the estimate).
    res = g - cv2.medianBlur(Gp, 3).astype(np.float32)
    sm = cv2.GaussianBlur(g, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
    flat = block_view((grad < 25).astype(np.float32), B) > 0.5
    rv = np.where(flat, block_view(res, B), np.nan).reshape(h, w, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(rv, axis=-1)
        mad = np.nanmedian(np.abs(rv - med[..., None]), axis=-1)
    sigma = 1.4826 * mad
    few = flat.reshape(h, w, -1).mean(-1) < 0.25
    sigma[few | np.isnan(sigma)] = np.nanmedian(sigma[~few]) if (~few).any() else 0.0
    F["sigma"] = sigma.astype(np.float32)

    # Texture: LBP histogram (uniform, P=8, R=1) + GLCM contrast / homogeneity
    lbp = local_binary_pattern(Gp, 8, 1, "uniform")
    lb = block_view(lbp, B)
    F["lbp"] = np.stack([(lb == k).mean((2, 3)) for k in range(10)], -1).astype(np.float32)
    q = (Gp // 16).astype(np.uint8)
    con = np.zeros((h, w), np.float32)
    hom = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            P = graycomatrix(q[i * B:(i + 1) * B, j * B:(j + 1) * B], [1], [0, np.pi / 2],
                             levels=16, symmetric=True, normed=True)
            con[i, j] = graycoprops(P, "contrast").mean()
            hom[i, j] = graycoprops(P, "homogeneity").mean()
    F["contrast"], F["homog"] = np.log1p(con), hom
    return F


# --------------------------------------------------------------------------- #
# Analyzer 1: Error Level Analysis      E = |I - JPEG_q(I)|
# --------------------------------------------------------------------------- #
def analyze_ela(I0, q, size_hw, B, cls):
    ok, enc = cv2.imencode(".jpg", I0, [cv2.IMWRITE_JPEG_QUALITY, q])
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    E = np.abs(I0.astype(np.float32) - rec.astype(np.float32)).mean(-1)
    H, W = size_hw
    E = cv2.resize(E, (W, H), interpolation=cv2.INTER_AREA)
    Ep = pad_to_multiple(E, B)
    grid = block_view(Ep, B).mean((2, 3))
    return class_z(grid, cls, floor=0.3)


# --------------------------------------------------------------------------- #
# Analyzer 2: Copy-move       ORB + ratio test (d1/d2 < 0.75) + RANSAC
# --------------------------------------------------------------------------- #
def analyze_copymove(G, cfg: Config):
    H, W = G.shape
    out = np.zeros((H, W), np.float32)
    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    kps, des = orb.detectAndCompute(G, None)
    if des is None or len(kps) < 30:
        return out, 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=4)
    src, dst = [], []
    for m in knn:
        cand = [x for x in m if x.trainIdx != x.queryIdx]   # drop the self-match
        if len(cand) < 2:
            continue
        a, b = cand[0], cand[1]
        if a.distance < cfg.cm_ratio * b.distance:
            p, q = kps[a.queryIdx].pt, kps[a.trainIdx].pt
            if np.hypot(p[0] - q[0], p[1] - q[1]) > cfg.cm_min_shift:
                src.append(p)
                dst.append(q)
    if len(src) < cfg.cm_min_inliers:
        return out, 0
    src, dst = np.float32(src), np.float32(dst)
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if inl is None or int(inl.sum()) < cfg.cm_min_inliers:
        return out, 0
    inl = inl.ravel().astype(bool)
    for pt in np.vstack([src[inl], dst[inl]]):
        cv2.circle(out, (int(pt[0]), int(pt[1])), 22, 1.0, -1)
    out = cv2.GaussianBlur(out, (0, 0), 10)
    return out / max(out.max(), 1e-6), int(inl.sum())


# --------------------------------------------------------------------------- #
# Analyzer 3: Noise residual
# --------------------------------------------------------------------------- #
def analyze_noise(F):
    return class_z(F["sigma"], F["cls"], floor=1.0)


# --------------------------------------------------------------------------- #
# Analyzer 4: Texture (LBP + GLCM)
# --------------------------------------------------------------------------- #
def analyze_texture(F):
    cls = F["cls"]
    d = np.zeros(cls.shape, np.float32)
    for c in (0, 1):
        m = cls == c
        if m.sum() >= 15:
            d[m] = np.abs(F["lbp"][m] - np.median(F["lbp"][m], axis=0)).sum(-1)
    z1 = class_z(d, cls, floor=0.05)
    z2 = class_z(F["contrast"], cls, floor=0.15)
    z3 = class_z(F["homog"], cls, floor=0.03)
    return (z1 + z2 + z3) / 3


# --------------------------------------------------------------------------- #
# Analyzer 5: Frequency (8x8 block DCT + local FFT)
# --------------------------------------------------------------------------- #
def analyze_freq(Gp, B, cls):
    h, w = cls.shape
    g = Gp.astype(np.float32) - 128
    H, W = g.shape
    # DCT on the JPEG 8x8 grid: share of high-frequency AC energy
    bl = g.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    C = np.abs(dctn(bl, axes=(2, 3), norm="ortho"))
    u, v = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    ac = C.sum((2, 3)) - C[:, :, 0, 0]
    hf = (C * (u + v >= 7)).sum((2, 3)) / (ac + 1.0)
    r = B // 8
    hf = hf[:h * r, :w * r].reshape(h, r, w, r).mean((1, 3))
    # FFT: S = log(1 + |F(u,v)|), fraction of energy at high radius
    blk = block_view(g, B)
    win = np.outer(np.hanning(B), np.hanning(B)).astype(np.float32)
    Fm = np.abs(np.fft.fftshift(np.fft.fft2((blk - blk.mean((2, 3), keepdims=True)) * win), axes=(2, 3)))
    S = np.log1p(Fm)
    yy, xx = np.mgrid[:B, :B]
    outer = np.hypot(yy - B / 2, xx - B / 2) > 0.25 * B
    fft_hf = (S * outer).sum((2, 3)) / (S.sum((2, 3)) + 1e-6)
    return (class_z(hf, cls, 0.02) + class_z(fft_hf, cls, 0.02)) / 2


# --------------------------------------------------------------------------- #
# Analyzer 6: Splicing   D_b = ||f_b - f_ring(b)|| / s, needs >= 2 cues to agree
# --------------------------------------------------------------------------- #
_RING = np.ones((5, 5), np.float32)
_RING[1:4, 1:4] = 0          # ring = blocks at distance 2 (inner 3x3 excluded)


def _ring_mean(f, cls):
    out = f.astype(np.float32).copy()
    for c in (0, 1):
        m = (cls == c).astype(np.float32)
        num = cv2.filter2D(f.astype(np.float32) * m, -1, _RING, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(m, -1, _RING, borderType=cv2.BORDER_CONSTANT)
        ok = (cls == c) & (den >= 3)
        out[ok] = num[ok] / den[ok]
    return out


def analyze_splicing(F):
    cls = F["cls"]
    d = {k: np.abs(F[k] - _ring_mean(F[k], cls)) for k in ("sigma", "mean", "std", "contrast")}
    lbp_d = np.zeros(cls.shape, np.float32)
    for k in range(10):
        lbp_d += np.abs(F["lbp"][..., k] - _ring_mean(F["lbp"][..., k], cls))
    d["lbp"] = lbp_d
    floors = {"sigma": 1.0, "mean": 3.0, "std": 2.0, "contrast": 0.2, "lbp": 0.08}
    Z = []
    for k, x in d.items():
        med = np.median(x)
        s = max(med + 1.4826 * np.median(np.abs(x - med)), floors[k])
        Z.append(np.clip(x / s, 0, 10))
    Z = np.stack(Z, -1)
    agree = (Z > 3).sum(-1) >= 2
    return (Z.mean(-1) * agree).astype(np.float32)


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
def localize(S, cfg: Config):
    H, W = S.shape
    Ss = cv2.GaussianBlur(S, (0, 0), cfg.block / 2)
    otsu, _ = cv2.threshold(np.clip(Ss * 255, 0, 255).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tau = max(otsu / 255.0, cfg.min_tau)
    mask = ((Ss > tau) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes, clean = [], np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= cfg.min_area_frac * H * W:
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                          "strength": float(Ss[labels == i].mean())})
            clean[labels == i] = 255
    boxes.sort(key=lambda b: -b["strength"])
    return clean, boxes, tau


def make_overlay(I1, S, boxes):
    heat = cv2.applyColorMap((np.clip(S, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(I1, 0.6, heat, 0.4, 0)
    ann = I1.copy()
    for b in boxes:
        cv2.rectangle(ann, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 0, 255), 3)
    return heat, over, ann


# --------------------------------------------------------------------------- #
# Master algorithm: DetectForgery(I)
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    score: float
    verdict: str
    S: np.ndarray
    mask: np.ndarray
    boxes: list
    cue_maps: dict
    heatmap: np.ndarray
    overlay: np.ndarray
    annotated: np.ndarray
    info: dict


def detect_forgery(I0: np.ndarray, I1: np.ndarray | None = None, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    if I1 is None:
        s = cfg.max_side / max(I0.shape[:2])
        I1 = cv2.resize(I0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else I0.copy()
    B = cfg.block
    G = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    H, W = G.shape
    Gp = pad_to_multiple(G, B)
    F = block_features(Gp, B, cfg.blank_std)

    # lines 03-08: six analyzers, each returns a suspicion map
    maps = {}
    maps["ela"] = grid_to_full(analyze_ela(I0, cfg.ela_quality, (H, W), B, F["cls"]), B, H, W)
    cm, n_inl = analyze_copymove(G, cfg)
    maps["copymove"] = cm
    maps["noise"] = grid_to_full(analyze_noise(F), B, H, W)
    maps["texture"] = grid_to_full(analyze_texture(F), B, H, W)
    maps["freq"] = grid_to_full(analyze_freq(Gp, B, F["cls"]), B, H, W)
    maps["splicing"] = grid_to_full(analyze_splicing(F), B, H, W)

    # lines 09-11: normalise, fuse, score
    norm = {k: robust_normalize(m, cfg.norm_floor[k]) for k, m in maps.items()}
    S = sum(cfg.weights[k] * norm[k] for k in norm).astype(np.float32)
    k = max(1, int(cfg.top_frac * S.size))
    top = float(np.partition(S.ravel(), -k)[-k:].mean())
    score = float(np.clip(100 * cfg.score_gain * top, 0, 100))

    # line 12-13: localise and decide
    mask, boxes, tau = localize(S, cfg)
    if score < cfg.thr_review:
        verdict, boxes = "AUTHENTIC", []
    elif score < cfg.thr_forged:
        verdict = "FLAG FOR REVIEW"
    else:
        verdict = "POTENTIALLY FORGED"
    heat, over, ann = make_overlay(I1, S, boxes)
    return Result(score, verdict, S, mask, boxes, norm, heat, over, ann,
                  {"tau": tau, "copymove_inliers": n_inl, "image_size": (W, H)})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Document forgery detection & localization")
    ap.add_argument("image")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config()
    I0, I1 = load_image(args.image, cfg)
    r = detect_forgery(I0, I1, cfg)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_heatmap.png"), r.overlay)
    cv2.imwrite(os.path.join(args.out, f"{stem}_boxes.png"), r.annotated)
    for name, m in r.cue_maps.items():
        cv2.imwrite(os.path.join(args.out, f"{stem}_cue_{name}.png"), (m * 255).astype(np.uint8))
    report = {"file": args.image, "score": round(r.score, 1), "verdict": r.verdict,
              "boxes": r.boxes, **{k: v for k, v in r.info.items() if k != "image_size"}}
    with open(os.path.join(args.out, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{r.verdict}  |  evidence score {r.score:.1f}/100  |  {len(r.boxes)} suspicious region(s)")
    print(f"Outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()
