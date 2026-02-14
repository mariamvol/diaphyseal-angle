from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from .models import pred2_to_deg


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _to_int_box_xyxy(box, W, H):
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    if hasattr(box, "tolist"):
        box = box.tolist()
    x1, y1, x2, y2 = box
    x1 = int(round(float(x1))); y1 = int(round(float(y1)))
    x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = int(max(0, min(W-1, x1)))
    y1 = int(max(0, min(H-1, y1)))
    x2 = int(max(0, min(W-1, x2)))
    y2 = int(max(0, min(H-1, y2)))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2


def obtuse_from_acute(a_deg: float) -> float:
    a = float(a_deg)
    a = max(0.0, min(90.0, a))
    return 180.0 - a


def nsa_status_obtuse(nsa_obt: float, normal=(125, 145), border=5) -> str:
    lo, hi = normal
    if lo <= nsa_obt <= hi:
        return "NORMAL"
    if (lo - border) <= nsa_obt < lo or hi < nsa_obt <= (hi + border):
        return "BORDERLINE"
    return "ABNORMAL"


def draw_info_box(img_rgb, x, y, lines, font_scale=1.0, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 10
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    h = sum(s[1] for s in sizes) + pad * (len(lines) + 1)

    H, W = img_rgb.shape[:2]
    x = int(max(0, min(x, W - w - 1)))
    y = int(max(0, min(y, H - h - 1)))

    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 0), -1)

    cy = y + pad + sizes[0][1]
    for t, s in zip(lines, sizes):
        cv2.putText(img_rgb, t, (x + pad, cy), font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        cy += s[1] + pad

    return img_rgb


@dataclass
class PredictResult:
    nsa_acute_deg: float
    nsa_obtuse_deg: float
    bbox_xyxy: List[int]
    det_score: float
    status: str


def predict_nsa_full_image(
    img_path: str,
    det_model,
    reg_model,
    device: str = "cpu",
    det_img_max: int = 1024,
    crop_size: int = 384,
    score_thr: float = 0.3,
    normal_range=(125, 145),
    border: int = 5,
) -> Optional[PredictResult]:
    det_model.eval()
    reg_model.eval()

    img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(img_path)

    H0, W0 = img0.shape[:2]
    img_rgb0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)

    # detector resize
    scale = 1.0
    if max(H0, W0) > det_img_max:
        scale = det_img_max / max(H0, W0)
        newW, newH = int(round(W0 * scale)), int(round(H0 * scale))
        img_det = cv2.resize(img_rgb0, (newW, newH), interpolation=cv2.INTER_AREA)
    else:
        img_det = img_rgb0

    x_det = TF.to_tensor(img_det).to(device)

    with torch.no_grad():
        pred = det_model([x_det])[0]

    if len(pred["boxes"]) == 0:
        return None

    j = int(torch.argmax(pred["scores"]).item())
    score = float(pred["scores"][j].item())
    if score < score_thr:
        return None

    box_scaled = pred["boxes"][j].detach().cpu().numpy().tolist()
    box = [b / scale for b in box_scaled]  # back to original coords

    x1, y1, x2, y2 = _to_int_box_xyxy(box, W0, H0)

    crop = img0[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    crop_rs = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    crop_rs_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_GRAY2RGB)

    x = TF.to_tensor(crop_rs_rgb)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y2_pred = reg_model(x)
        nsa_acute = float(pred2_to_deg(y2_pred).item())

    nsa_obt = obtuse_from_acute(nsa_acute)
    status = nsa_status_obtuse(nsa_obt, normal=normal_range, border=border)

    return PredictResult(
        nsa_acute_deg=nsa_acute,
        nsa_obtuse_deg=nsa_obt,
        bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)],
        det_score=score,
        status=status,
    )


def visualize_prediction(
    img_path: str,
    result: PredictResult,
    out_path: str,
    normal_range=(125, 145),
):
    img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(img_path)
    vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)

    x1, y1, x2, y2 = result.bbox_xyxy
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 4)

    lines = [
        f"Pred obtuse: {result.nsa_obtuse_deg:.1f} deg   (acute {result.nsa_acute_deg:.1f})",
        f"status: {result.status}   norm {normal_range[0]}-{normal_range[1]}",
        f"det_score: {result.det_score:.3f}",
    ]

    tx, ty = x1, y2 + 10
    H0, W0 = vis.shape[:2]
    if ty + 140 > H0:
        ty = max(0, y1 - 150)

    vis = draw_info_box(vis, tx, ty, lines, font_scale=1.0, thickness=2)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
