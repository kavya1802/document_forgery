# Document Forgery Detection & Localization

**Multi-domain image analysis using classical Digital Image Processing** — no deep learning, no training, runs on CPU.

The system takes a scanned or photographed document, runs six independent forensic analyzers, fuses their evidence into a single suspicion map, and outputs a verdict, an evidence score (0–100), and bounding boxes around suspected tampered regions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/document-forgery-detection/blob/main/Document_Forgery_Detection.ipynb)

## Pipeline

```
Input → Preprocess (resize, grayscale, 32×32 blocks)
      → 6 analyzers ─┬─ Error Level Analysis (JPEG re-compression, q = 90)
                     ├─ Copy-move (ORB + ratio test + RANSAC)
                     ├─ Noise residual (MAD of median-filter residual)
                     ├─ Texture (LBP histograms + GLCM contrast/homogeneity)
                     ├─ Frequency (8×8 block DCT + local FFT)
                     └─ Splicing (block vs. surrounding-ring feature inconsistency)
      → Robust normalisation + weighted fusion
      → Otsu thresholding + morphology → bounding boxes
      → Verdict: AUTHENTIC (<30) · FLAG FOR REVIEW (30–60) · POTENTIALLY FORGED (≥60)
```

| Cue | Weight |
|---|---|
| Copy-move | 0.25 |
| ELA | 0.20 |
| Noise | 0.15 |
| Frequency (FFT/DCT) | 0.15 |
| Splicing | 0.15 |
| Texture | 0.10 |

Blocks are split into *blank paper* and *content* classes and each cue is scored with a robust z-score within its own class, so text is compared with text and background with background.

## Repository contents

| File | Purpose |
|---|---|
| `Document_Forgery_Detection.ipynb` | Colab notebook: demo, upload-your-own image, per-cue maps, benchmark |
| `forgery_detector.py` | Detection engine + command-line tool |
| `synthetic.py` | Synthetic forged-document generator and evaluation (Accuracy, Precision, Recall, F1, IoU, Dice) |
| `requirements.txt` | Python dependencies |

## Quick start

**Colab:** click the badge above, then *Runtime → Run all*.

**Locally:**
```bash
pip install -r requirements.txt

# analyse one document
python forgery_detector.py path/to/document.jpg --out results/

# synthetic benchmark (each set = 1 authentic + 3 forged pages)
python synthetic.py --n 10

# save sample forged images with ground-truth masks
python synthetic.py --save samples/

# evaluate on a public dataset (CoMoFoD / CASIA v2.0 / DocTamper layout)
python synthetic.py --images path/to/images --masks path/to/masks
```

> The synthetic generator uses DejaVu fonts (`sudo apt install fonts-dejavu-core` on Linux). Without them it falls back to Pillow's default font.

Outputs per image: `*_heatmap.png` (fused heatmap overlay), `*_boxes.png` (localised regions), `*_cue_<name>.png` (each analyzer's map) and `*_report.json` (score, verdict, boxes).

## Forgery types covered by the synthetic benchmark

- **Copy-move** — a region of the page duplicated elsewhere
- **Splicing** — a region pasted from a document with different noise and JPEG history
- **Re-typed text** — a line erased and retyped in a different font

## Limitations

Classical cues depend on compression and noise traces, so heavily re-compressed, resized or re-scanned images weaken ELA and noise evidence. Scores are calibrated on the synthetic set; thresholds may need tuning for other document sources.

## Authors

- Kavya M Thakker
- Chiranth

Digital Image Processing major project — SRM Institute of Science and Technology.
