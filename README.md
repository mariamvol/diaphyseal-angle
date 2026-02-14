# Diaphyseal Angle (NSA) — inference only

End-to-end prediction of **Neck–Shaft Angle (NSA)** from a full X-ray:

1) **ROI detector**: Faster R-CNN (ResNet50-FPN) finds the shoulder ROI  
2) **Angle regressor**: ResNet18 predicts `[cos(2θ), sin(2θ)]` → converts to **acute angle** `θ∈[0,90]`  
3) Converts to **obtuse NSA**: `NSA_obtuse = 180 - θ`  
4) Optional visualization: bbox + text overlay

## Requirements

- Python 3.9+ recommended
- CPU works, CUDA optional

Install deps:
```bash
pip install -r requirements.txt
```

## Repo structure (important)

Your repo should contain these files:
```
diaphyseal_angle/
  __init__.py
  models.py
  predict.py
  infer_one.py
requirements.txt
README.md
```

## Weights

You need two weight files:

- `roi_detector_best.pt`  (detector)
- `nsa_regressor_best.pt` (regressor)

Locally you can place them like:
```
weights/roi_detector_best.pt
weights/nsa_regressor_best.pt
```

## Run inference on one image

```bash
python -m diaphyseal_angle.infer_one   --img xray.jpg   --det weights/roi_detector_best.pt   --reg weights/nsa_regressor_best.pt   --out_json result.json   --out_vis vis.png
```

### Useful flags

- `--device cpu|cuda` (default: auto)
- `--score_thr 0.3` detector score threshold
- `--det_img_max 1024` max side for detector input resize
- `--crop_size 384` crop size to regressor
- `--normal_lo 125 --normal_hi 145 --border 5` for status thresholds

## Output

Console prints something like:
- `NSA acute:  37.42 deg`
- `NSA obtuse: 142.58 deg`
- `status:     NORMAL`
- `det_score:  0.812`
- `bbox_xyxy:  [x1, y1, x2, y2]`

### JSON (`--out_json`)

```json
{
  "ok": true,
  "nsa_acute_deg": 37.42,
  "nsa_obtuse_deg": 142.58,
  "status": "NORMAL",
  "det_score": 0.812,
  "bbox_xyxy": [123, 88, 612, 701],
  "normal_range": [125.0, 145.0],
  "border": 5,
  "device": "cuda",
  "out_vis": "vis.png"
}
```

If detection fails / low score:
```json
{"ok": false, "reason": "no_detection_or_low_score"}
```

## Notes / gotchas

- Input image is read as grayscale, converted to RGB internally.
- If your X-rays are huge, detector input is resized (default max side = 1024).
- The regressor always predicts **acute** angle in `[0, 90]`, then we convert to obtuse NSA.

---
## Google Colab Quick Start
Run the full NSA inference pipeline in Google Colab.

###Clone repository
```python
!git clone https://github.com/YOUR_USERNAME/diaphyseal-angle.git
%cd diaphyseal-angle
```
###Install dependencies
```python
!pip install -r requirements.txte
```
###Download model weights (from Releases)
```python
!wget https://github.com/mariamvol/diaphyseal-angle/releases/download/v0.1.0/roi_detector_best.pt
!wget https://github.com/mariamvol/diaphyseal-angle/releases/download/v0.1.0/nsa_regressor_best.pt
```
###Upload an X-ray image
```python
from google.colab import files
files.upload()  # upload xray.jpg
```
###Run inference
```python
!python -m diaphyseal_angle.infer_one \
    --img xray.jpg \
    --det roi_detector_best.pt \
    --reg nsa_regressor_best.pt \
    --out_json result.json \
    --out_vis vis.png
```
###Output
- `result.json` — numeric results
- `vis.png` — visualization with bbox and predicted NSA
---
## License

MIT
