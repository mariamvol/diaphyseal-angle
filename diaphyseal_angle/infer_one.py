import argparse
import json
import torch

from .models import load_detector_weights, load_regressor_weights
from .predict import predict_nsa_full_image, visualize_prediction


def main():
    p = argparse.ArgumentParser("Diaphyseal angle (NSA) inference on one X-ray")
    p.add_argument("--img", required=True, help="Path to input X-ray image (jpg/png)")
    p.add_argument("--det", required=True, help="Path to detector weights .pt (roi_detector_best.pt)")
    p.add_argument("--reg", required=True, help="Path to regressor weights .pt (nsa_regressor_best.pt)")
    p.add_argument("--device", default=None, help="cpu or cuda. default: auto")
    p.add_argument("--score_thr", type=float, default=0.3, help="detector score threshold")
    p.add_argument("--det_img_max", type=int, default=1024, help="max side for detector input")
    p.add_argument("--crop_size", type=int, default=384, help="crop size for regressor")
    p.add_argument("--normal_lo", type=float, default=125.0)
    p.add_argument("--normal_hi", type=float, default=145.0)
    p.add_argument("--border", type=int, default=5)
    p.add_argument("--out_json", default=None, help="save result json")
    p.add_argument("--out_vis", default=None, help="save visualization image")
    args = p.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    det_model = load_detector_weights(args.det, map_location="cpu").to(device)
    reg_model = load_regressor_weights(args.reg, map_location="cpu").to(device)

    res = predict_nsa_full_image(
        img_path=args.img,
        det_model=det_model,
        reg_model=reg_model,
        device=device,
        det_img_max=args.det_img_max,
        crop_size=args.crop_size,
        score_thr=args.score_thr,
        normal_range=(args.normal_lo, args.normal_hi),
        border=args.border,
    )

    if res is None:
        print("NO_PREDICTION (no detection / score<thr / empty crop)")
        payload = {"ok": False, "reason": "no_detection_or_low_score"}
    else:
        print(f"NSA acute:  {res.nsa_acute_deg:.2f} deg")
        print(f"NSA obtuse: {res.nsa_obtuse_deg:.2f} deg")
        print(f"status:     {res.status}")
        print(f"det_score:  {res.det_score:.3f}")
        print(f"bbox_xyxy:  {res.bbox_xyxy}")

        payload = {
            "ok": True,
            "nsa_acute_deg": res.nsa_acute_deg,
            "nsa_obtuse_deg": res.nsa_obtuse_deg,
            "status": res.status,
            "det_score": res.det_score,
            "bbox_xyxy": res.bbox_xyxy,
            "normal_range": [args.normal_lo, args.normal_hi],
            "border": args.border,
            "device": device,
        }

        if args.out_vis:
            visualize_prediction(
                img_path=args.img,
                result=res,
                out_path=args.out_vis,
                normal_range=(args.normal_lo, args.normal_hi),
            )
            payload["out_vis"] = args.out_vis

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
