"""Classical (non-neural) super-resolution baselines: nearest, bilinear, bicubic, Lanczos.

Every learned model in this project has to beat plain interpolation to be worth its
weights. Without this row in the table a PSNR of 26 dB means nothing - it could be below
what `Image.resize` gives for free.

Bicubic is the conventional reference in the super-resolution literature, so it is the
one to quote; the others are here to show the spread.

Protocol matches how the learned models are evaluated: take the ground-truth image,
downscale by `scale` to make the low-resolution input, upscale back, and compare against
the original. Metrics are PSNR and SSIM on the Y (luma) channel, which is also the
convention - RGB PSNR flatters every method equally and hides chroma error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

FILTERS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def to_y(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma in [16, 235], the convention for SR metrics.

    The coefficients assume R,G,B in **[0, 1]**, not [0, 255]. Feeding 8-bit values
    straight in gives Y in the tens of thousands, and PSNR against a 255 peak then comes
    out *negative* - which is physically impossible and was the tell that this was
    wrong, not that interpolation had somehow failed.
    """
    x = img / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def psnr(a: np.ndarray, b: np.ndarray, border: int = 4) -> float:
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, border: int = 4) -> float:
    """Global SSIM with an 11x11 uniform window, no scipy dependency."""
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = 11
    def blur(x):
        c = np.cumsum(np.cumsum(np.pad(x, ((k, k), (k, k)), mode="edge"), 0), 1)
        s = (c[k * 2:, k * 2:] - c[:-k * 2, k * 2:]
             - c[k * 2:, :-k * 2] + c[:-k * 2, :-k * 2])
        return s / (k * 2) ** 2
    mu_a, mu_b = blur(a), blur(b)
    saa, sbb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "real-esrgan", "data", "gt"))
    ap.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "real-esrgan", "data", "meta_info", "test.txt"))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results"))
    args = ap.parse_args()

    if os.path.exists(args.split):
        with open(args.split, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        files = [os.path.join(args.gt, n) for n in names]
    else:
        files = sorted(glob.glob(os.path.join(args.gt, "*.png")))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print(f"  no images found under {args.gt}")
        return

    scores = {k: {"psnr": [], "ssim": []} for k in FILTERS}
    for f in files:
        hr = Image.open(f).convert("RGB")
        # crop so the size divides by scale exactly - otherwise the round trip
        # introduces a resampling error that is not the method's fault
        w, h = (hr.width // args.scale) * args.scale, (hr.height // args.scale) * args.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // args.scale, h // args.scale), Image.BICUBIC)
        y_hr = to_y(np.asarray(hr, dtype=np.float64))
        for name, filt in FILTERS.items():
            sr = lr.resize((w, h), filt)
            y_sr = to_y(np.asarray(sr, dtype=np.float64))
            scores[name]["psnr"].append(psnr(y_hr, y_sr))
            scores[name]["ssim"].append(ssim(y_hr, y_sr))

    os.makedirs(args.out, exist_ok=True)
    print(f"  {len(files)} images, x{args.scale}, Y-channel, 4px border cropped\n")
    print(f"  {'method':10s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    summary = {}
    for name in FILTERS:
        p = float(np.mean(scores[name]["psnr"]))
        s = float(np.mean(scores[name]["ssim"]))
        summary[name] = {"psnr": round(p, 4), "ssim": round(s, 4),
                         "n": len(scores[name]["psnr"])}
        mark = "  <- reference" if name == "bicubic" else ""
        print(f"  {name:10s} {p:10.2f} {s:8.4f}{mark}")

    with open(os.path.join(args.out, "baseline_classical.json"), "w", encoding="utf-8") as fh:
        json.dump({"scale": args.scale, "n_images": len(files),
                   "metric": "Y-channel, 4px border", "results": summary}, fh, indent=2)
    print(f"\n  wrote {os.path.join(args.out, 'baseline_classical.json')}")
    print("  Any learned model must beat the bicubic row to justify itself.")


if __name__ == "__main__":
    main()
