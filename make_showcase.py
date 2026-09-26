"""Run inference on 10 test images with each finished model, keep the best 2.

Produces, per model, a side-by-side panel: degraded input (what the model is given),
the model's output, and the original. Ten are scored, the two with the highest PSNR
gain over the bicubic baseline are kept for the repo.

Selecting on gain-over-bicubic rather than raw PSNR matters: a smooth image scores high
PSNR for everyone, so picking by raw score would just select the easiest images rather
than the ones where the model actually contributes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "basicsr", "BasicSR"))
from baseline_classical import psnr, ssim, to_y  # noqa: E402
from train_sr import DATA, build  # noqa: E402

OUT = os.path.join(HERE, "showcase")
MODELS = ["msrresnet", "msrgan"]
SCALE = 4


def font(sz=17):
    for n in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(items, labels, out_path, pad=10, header=30):
    w, h = items[0].size
    W = len(items) * w + (len(items) + 1) * pad
    H = h + 2 * pad + header
    c = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(c)
    f = font()
    for i, (im, lb) in enumerate(zip(items, labels)):
        x = pad + i * (w + pad)
        c.paste(im, (x, header))
        d.text((x + 4, 7), lb, fill=(20, 20, 20), font=f)
    c.save(out_path)


@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(DATA, "meta_info", "test.txt"), encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()][:10]
    os.makedirs(OUT, exist_ok=True)

    summary = {}
    for mname in MODELS:
        ck = os.path.join(HERE, "results", mname, f"{mname}_best.pth")
        if not os.path.exists(ck):
            print(f"  skip {mname}: no checkpoint"); continue
        model, _ = build(mname, device)
        model.load_state_dict(torch.load(ck, map_location=device)["params"], strict=True)
        model.eval()

        scored = []
        for n in names:
            hr = Image.open(os.path.join(DATA, "gt", n)).convert("RGB")
            lr = Image.open(os.path.join(DATA, "lr_x4", n)).convert("RGB")
            x = torch.from_numpy(np.asarray(lr, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
            y = model(x).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
            sr = Image.fromarray((y * 255).round().astype("uint8"))
            if sr.size != hr.size:
                sr = sr.resize(hr.size, Image.BICUBIC)
            bic = lr.resize(hr.size, Image.BICUBIC)
            a = to_y(np.asarray(hr, np.float64))
            p_sr = psnr(a, to_y(np.asarray(sr, np.float64)))
            p_bi = psnr(a, to_y(np.asarray(bic, np.float64)))
            s_sr = ssim(a, to_y(np.asarray(sr, np.float64)))
            scored.append({"name": n, "psnr": p_sr, "bicubic": p_bi,
                           "gain": p_sr - p_bi, "ssim": s_sr,
                           "_im": (lr, sr, hr, bic)})

        # keep the two where the model contributes most over free interpolation
        scored.sort(key=lambda r: -r["gain"])
        keep = scored[:2]
        md = os.path.join(OUT, mname)
        os.makedirs(md, exist_ok=True)
        for rank, r in enumerate(keep, 1):
            lr, sr, hr, bic = r.pop("_im")
            stem = os.path.splitext(r["name"])[0]
            lr_up = lr.resize(hr.size, Image.NEAREST)     # show the actual input size
            panel([lr_up, sr, hr],
                  [f"Input (low-res {lr.size[0]}x{lr.size[1]})",
                   f"{mname} output  {r['psnr']:.2f} dB",
                   f"Original  ({hr.size[0]}x{hr.size[1]})"],
                  os.path.join(md, f"{rank:02d}_{stem}_comparison.png"))
            lr_up.save(os.path.join(md, f"{rank:02d}_{stem}_input.png"))
            sr.save(os.path.join(md, f"{rank:02d}_{stem}_output.png"))
            hr.save(os.path.join(md, f"{rank:02d}_{stem}_original.png"))
        for r in scored:
            r.pop("_im", None)
        summary[mname] = {"scored": [{k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in r.items()} for r in scored],
                          "kept": [r["name"] for r in keep]}
        print(f"  {mname}: scored {len(scored)}, kept {[r['name'] for r in keep]} "
              f"(gain {keep[0]['gain']:+.2f}, {keep[1]['gain']:+.2f} dB)")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "showcase.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
