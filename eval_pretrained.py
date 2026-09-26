"""Zero-shot PSNR/SSIM for every super-resolution model, before any fine-tuning.

No training involved. Each published checkpoint is run as-is on the same 50 test images
used by the classical baseline, so the numbers sit in one table with bicubic.

This is worth having on its own: it says what the off-the-shelf models achieve on *this*
data, which is the number fine-tuning has to beat. Without it, a fine-tuned result has
nothing to be compared against except the paper's numbers, which were measured on a
different dataset.

Protocol is identical to baseline_classical.py - downscale the ground truth by 4 with
bicubic to make the input, upscale back, compare on the Y channel with a 4px border
crop. Same protocol for every row or the comparison is meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline_classical import psnr, ssim, to_y  # noqa: E402

WEIGHTS = os.path.join(HERE, "..", "_weights")


def build_models(device: str) -> dict:
    """Each entry: (model, param_key, label). Missing checkpoints are skipped, loudly."""
    out = {}
    sys.path.insert(0, os.path.join(HERE, "basicsr", "BasicSR"))
    sys.path.insert(0, os.path.join(HERE, "real-esrgan", "Real-ESRGAN"))

    def load(model, path, keys=("params_ema", "params")):
        if not os.path.exists(path):
            print(f"    skip (missing): {os.path.basename(path)}")
            return None
        sd = torch.load(path, map_location="cpu", weights_only=False)
        state = next((sd[k] for k in keys if isinstance(sd, dict) and k in sd), sd)
        model.load_state_dict(state, strict=True)   # strict: a silent mismatch is worse
        return model.eval().to(device)

    try:
        from basicsr.archs.srresnet_arch import MSRResNet
        m = load(MSRResNet(3, 3, 64, 16, 4), os.path.join(WEIGHTS, "msrresnet_x4_g.pth"))
        if m: out["MSRResNet"] = m
        m = load(MSRResNet(3, 3, 64, 16, 4), os.path.join(WEIGHTS, "msrgan_x4_g.pth"))
        if m: out["MSRGAN (SRGAN)"] = m
    except Exception as e:
        print(f"    BasicSR archs unavailable: {str(e)[:100]}")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        m = load(RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32),
                 os.path.join(WEIGHTS, "RealESRGAN_x4plus_anime_6B.pth"))
        if m: out["Real-ESRGAN anime_6B"] = m
        m = load(RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
                 os.path.join(WEIGHTS, "esrgan_x4_df2kost.pth"))
        if m: out["ESRGAN"] = m
        m = load(RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
                 os.path.join(WEIGHTS, "RealESRNet_x4plus.pth"))
        if m: out["RealESRNet"] = m
    except Exception as e:
        print(f"    RRDBNet unavailable: {str(e)[:100]}")

    return out


@torch.no_grad()
def run(model, lr_img: Image.Image, device: str) -> Image.Image:
    x = torch.from_numpy(
        np.asarray(lr_img, dtype=np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
    y = model(x).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((y * 255).round().astype("uint8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "real-esrgan", "data", "gt"))
    ap.add_argument("--split", default=os.path.join(HERE, "real-esrgan", "data",
                                                    "meta_info", "test.txt"))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.split, encoding="utf-8") as fh:
        files = [os.path.join(args.gt, ln.strip()) for ln in fh if ln.strip()]
    files = [f for f in files if os.path.exists(f)]
    print(f"  {len(files)} test images on {device}")

    models = build_models(device)
    if not models:
        print("  no models loaded"); return
    print(f"  loaded: {', '.join(models)}\n")

    scores = {k: {"psnr": [], "ssim": []} for k in models}
    scores["bicubic"] = {"psnr": [], "ssim": []}
    times = {k: 0.0 for k in models}

    for f in files:
        hr = Image.open(f).convert("RGB")
        w = (hr.width // args.scale) * args.scale
        h = (hr.height // args.scale) * args.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // args.scale, h // args.scale), Image.BICUBIC)
        y_hr = to_y(np.asarray(hr, dtype=np.float64))

        bic = lr.resize((w, h), Image.BICUBIC)
        yb = to_y(np.asarray(bic, dtype=np.float64))
        scores["bicubic"]["psnr"].append(psnr(y_hr, yb))
        scores["bicubic"]["ssim"].append(ssim(y_hr, yb))

        for name, m in models.items():
            t = time.perf_counter()
            sr = run(m, lr, device)
            times[name] += time.perf_counter() - t
            if sr.size != (w, h):
                sr = sr.resize((w, h), Image.BICUBIC)
            ys = to_y(np.asarray(sr, dtype=np.float64))
            scores[name]["psnr"].append(psnr(y_hr, ys))
            scores[name]["ssim"].append(ssim(y_hr, ys))

    os.makedirs(args.out, exist_ok=True)
    order = ["bicubic"] + list(models)
    print(f"  {'model':24s} {'PSNR (dB)':>10s} {'SSIM':>8s} {'s/img':>8s}")
    summary = {}
    base = float(np.mean(scores["bicubic"]["psnr"]))
    for name in order:
        p = float(np.mean(scores[name]["psnr"]))
        s = float(np.mean(scores[name]["ssim"]))
        t = times.get(name, 0.0) / max(len(files), 1)
        summary[name] = {"psnr": round(p, 4), "ssim": round(s, 4),
                         "sec_per_image": round(t, 4), "vs_bicubic_db": round(p - base, 4)}
        flag = "" if name == "bicubic" else ("  BELOW BICUBIC" if p < base else f"  +{p-base:.2f} dB")
        print(f"  {name:24s} {p:10.2f} {s:8.4f} {t:8.3f}{flag}")

    with open(os.path.join(args.out, "eval_pretrained.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_images": len(files), "scale": args.scale, "device": device,
                   "metric": "Y-channel PSNR/SSIM, 4px border", "results": summary}, fh, indent=2)
    print(f"\n  wrote {os.path.join(args.out, 'eval_pretrained.json')}")


if __name__ == "__main__":
    main()
