"""Build the super-resolution dataset from DIV2K.

Replaces the LIP-ATR set. Those originals were 400x600, so a x4 downsample gave a
100x150 input - below the point where detail can be recovered rather than invented.
Every model converged to ~28.5 dB regardless of architecture, which was the dataset's
information ceiling showing through, not an architectural limit.

DIV2K is ~2040x1356. At x4 the input is ~510x339: eleven times the pixels, and the
dataset every published super-resolution paper uses, so the numbers become comparable
to their reported figures.

Protocol follows BasicSR and Real-ESRGAN: crop fixed sub-images from the HR originals
rather than resizing whole 2K frames. Resizing a 2K image down to a trainable size
would throw away exactly the detail the model is meant to learn to reconstruct - the
crops preserve native pixel scale.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ZIPS = os.path.join(HERE, "..", "_data", "div2k")
RAW = os.path.join(HERE, "..", "_data", "div2k", "raw")
DST = os.path.join(HERE, "real-esrgan", "data")
SUB = 480          # HR sub-image size; /4 gives a 120px LR patch
SCALE = 4


def unzip() -> int:
    os.makedirs(RAW, exist_ok=True)
    n = 0
    for z in sorted(glob.glob(os.path.join(ZIPS, "*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".png")]
                todo = [m for m in members
                        if not os.path.exists(os.path.join(RAW, os.path.basename(m)))]
                print(f"  {os.path.basename(z)}: {len(members)} images, {len(todo)} to extract",
                      flush=True)
                for m in todo:
                    with zf.open(m) as src, open(os.path.join(RAW, os.path.basename(m)), "wb") as out:
                        out.write(src.read())
                    n += 1
        except zipfile.BadZipFile:
            print(f"  {os.path.basename(z)}: not a complete zip yet, skipping")
    return n


def sharp(im: Image.Image) -> float:
    """Detail lost by a 2x round trip, on edge pixels only.

    A soft high-resolution target teaches the model to produce soft output, so the
    softest crops are rejected. Same filter that removed 104 of 1,104 LIP-ATR
    candidates.
    """
    g = np.asarray(im.convert("L").resize((256, 256), Image.LANCZOS), dtype=float)
    rt = np.asarray(Image.fromarray(g.astype("uint8")).resize((128, 128), Image.LANCZOS)
                    .resize((256, 256), Image.LANCZOS), dtype=float)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    edge = gx > np.percentile(gx, 90)
    return float(np.abs(g[:-1, :-1] - rt[:-1, :-1])[edge].mean()) if edge.sum() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=4, help="sub-images per source photo")
    ap.add_argument("--min-sharp", type=float, default=8.0)
    ap.add_argument("--max", type=int, default=3000)
    args = ap.parse_args()

    got = unzip()
    src = sorted(glob.glob(os.path.join(RAW, "*.png")))
    if not src:
        raise SystemExit(f"  no images extracted under {RAW}")
    print(f"  {len(src)} source images ({got} newly extracted)")
    print(f"  sample size: {Image.open(src[0]).size}")

    gt_dir = os.path.join(DST, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    rng = random.Random(3407)

    kept, soft = 0, 0
    for p in src:
        if kept >= args.max:
            break
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w < SUB or h < SUB:
            continue
        for _ in range(args.per_image):
            if kept >= args.max:
                break
            x = rng.randint(0, w - SUB)
            y = rng.randint(0, h - SUB)
            crop = im.crop((x, y, x + SUB, y + SUB))
            if sharp(crop) < args.min_sharp:
                soft += 1
                continue
            crop.save(os.path.join(gt_dir, f"{kept:05d}.png"))
            kept += 1

    print(f"  kept {kept} sub-images of {SUB}x{SUB}, rejected {soft} as too soft "
          f"({100*soft/max(kept+soft,1):.1f}%)")

    # 80/15/5, the house split policy
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.png")))
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, round(n * .05)); n_val = max(1, round(n * .15))
    parts = {"test": names[:n_test], "val": names[n_test:n_test + n_val],
             "train": names[n_test + n_val:]}
    mi = os.path.join(DST, "meta_info")
    os.makedirs(mi, exist_ok=True)
    for k, v in parts.items():
        with open(os.path.join(mi, f"{k}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(v))
    print("  splits: " + "  ".join(f"{k} {len(v)}" for k, v in parts.items()))

    # LR half, bicubic x4 - the standard protocol, now on a source large enough for it
    lr_dir = os.path.join(DST, f"lr_x{SCALE}")
    os.makedirs(lr_dir, exist_ok=True)
    for nm in names:
        hr = Image.open(os.path.join(gt_dir, nm))
        hr.resize((SUB // SCALE, SUB // SCALE), Image.BICUBIC).save(os.path.join(lr_dir, nm))
    print(f"  wrote {len(names)} LR images at {SUB//SCALE}x{SUB//SCALE}")

    with open(os.path.join(DST, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "DIV2K (ETH Zurich)", "sub_image": SUB, "scale": SCALE,
                   "n": len(names), "splits": {k: len(v) for k, v in parts.items()},
                   "min_sharp": args.min_sharp,
                   "note": "480x480 crops at native pixel scale; LR is 120x120. "
                           "Replaces a 400x600 source whose x4 input was 100x150."},
                  fh, indent=2)


if __name__ == "__main__":
    main()
