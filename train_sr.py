"""Fine-tune the super-resolution models on the 1,000-image set.

Parameters are DERIVED FROM THE DATASET SIZE, not copied from the authors' configs.
Those configs were written for DIV2K (800 images, but 1000k iterations on 8 GPUs); the
invariant that transfers is the number of gradient steps, not the epoch count.

    steps = epochs * N_train / batch

Budget here is 20,000 steps per model: enough for a pretrained network to adapt to a new
image domain, short of the point where 800 images start being memorised. At batch 16
over 800 images that is 50 steps/epoch, so 20,000 steps = 400 epochs.

Learning rate follows two rules together:
  * it is a FINE-TUNE, so start from ~1/2 of the authors' from-scratch rate
  * it scales with batch size (sqrt, for a fine-tune)
BasicSR trains MSRResNet at 2e-4 from scratch with batch 16; 1e-4 at the same batch is
the fine-tune equivalent.

Inference images are written for every run, before and after, so results can be shown
rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "basicsr", "BasicSR"))
from baseline_classical import psnr, ssim, to_y  # noqa: E402

DATA = os.path.join(HERE, "real-esrgan", "data")
WEIGHTS = os.path.join(HERE, "..", "_weights")


class PairSet(Dataset):
    """Random aligned LR/HR crops. gt_size is the HR patch; LR patch is gt_size/scale."""

    def __init__(self, split: str, gt_size: int = 128, scale: int = 4, train: bool = True):
        with open(os.path.join(DATA, "meta_info", f"{split}.txt"), encoding="utf-8") as fh:
            self.names = [ln.strip() for ln in fh if ln.strip()]
        self.gt_size, self.scale, self.train = gt_size, scale, train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        n = self.names[i]
        hr = Image.open(os.path.join(DATA, "gt", n)).convert("RGB")
        lr = Image.open(os.path.join(DATA, "lr_x4", n)).convert("RGB")
        s, g = self.scale, self.gt_size
        if self.train:
            lw, lh = lr.size
            lg = g // s
            x = random.randint(0, max(0, lw - lg))
            y = random.randint(0, max(0, lh - lg))
            lr = lr.crop((x, y, x + lg, y + lg))
            hr = hr.crop((x * s, y * s, (x + lg) * s, (y + lg) * s))
            if random.random() < 0.5:
                lr, hr = lr.transpose(Image.FLIP_LEFT_RIGHT), hr.transpose(Image.FLIP_LEFT_RIGHT)
        f = lambda im: torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        return f(lr), f(hr)


def build(name: str, device: str):
    """Returns (model, checkpoint_path). strict=True everywhere - a silent key mismatch
    loads nothing and looks like a weak model rather than a loading bug."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srresnet_arch import MSRResNet
    spec = {
        "msrresnet": (lambda: MSRResNet(3, 3, 64, 16, 4), "msrresnet_x4_g.pth"),
        "msrgan":    (lambda: MSRResNet(3, 3, 64, 16, 4), "msrgan_x4_g.pth"),
        "esrgan":    (lambda: RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                                      num_block=23, num_grow_ch=32), "esrgan_x4_df2kost.pth"),
        "realesrnet": (lambda: RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                                       num_block=23, num_grow_ch=32), "RealESRNet_x4plus.pth"),
        "realesrgan": (lambda: RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                                       num_block=6, num_grow_ch=32),
                       "RealESRGAN_x4plus_anime_6B.pth"),
    }[name]
    m = spec[0]()
    p = os.path.join(WEIGHTS, spec[1])
    sd = torch.load(p, map_location="cpu", weights_only=False)
    state = next((sd[k] for k in ("params_ema", "params") if isinstance(sd, dict) and k in sd), sd)
    m.load_state_dict(state, strict=True)
    return m.to(device), p


@torch.no_grad()
def evaluate(model, split: str, device: str, scale: int = 4) -> dict:
    model.eval()
    with open(os.path.join(DATA, "meta_info", f"{split}.txt"), encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    ps, ss = [], []
    for n in names:
        hr = Image.open(os.path.join(DATA, "gt", n)).convert("RGB")
        lr = Image.open(os.path.join(DATA, "lr_x4", n)).convert("RGB")
        x = torch.from_numpy(np.asarray(lr, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
        y = model(x).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        sr = Image.fromarray((y * 255).round().astype("uint8"))
        if sr.size != hr.size:
            sr = sr.resize(hr.size, Image.BICUBIC)
        a = to_y(np.asarray(hr, np.float64)); b = to_y(np.asarray(sr, np.float64))
        ps.append(psnr(a, b)); ss.append(ssim(a, b))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(names)}


@torch.no_grad()
def save_inference(model, device, out_dir: str, tag: str, k: int = 4) -> None:
    """Write input / output / ground-truth triples so results can be shown."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(DATA, "meta_info", "test.txt"), encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()][:k]
    model.eval()
    for n in names:
        stem = os.path.splitext(n)[0]
        hr = Image.open(os.path.join(DATA, "gt", n)).convert("RGB")
        lr = Image.open(os.path.join(DATA, "lr_x4", n)).convert("RGB")
        x = torch.from_numpy(np.asarray(lr, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
        y = model(x).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((y * 255).round().astype("uint8")).save(
            os.path.join(out_dir, f"{stem}_{tag}_output.png"))
        lr.resize(hr.size, Image.NEAREST).save(os.path.join(out_dir, f"{stem}_input_lr.png"))
        hr.save(os.path.join(out_dir, f"{stem}_ground_truth.png"))
        lr.resize(hr.size, Image.BICUBIC).save(os.path.join(out_dir, f"{stem}_bicubic.png"))


class LRAutoTune:
    """Check every N epochs whether learning has stalled, and fix the LR without asking.

    The failure this exists for: MSRGAN sat at 28.20 for 6,000 steps while its LR
    decayed from 1e-4 to 5e-5. Nothing was wrong with the loss, nothing raised, and the
    run looked healthy - it was simply learning too slowly to move. A schedule that only
    ever decays cannot recover from that.

    Rules, applied on the val metric:
      gain >= `good`          -> leave it alone, it is working
      0 < gain < `good`       -> too slow: DOUBLE the lr (up to `max_lr`)
      gain <= 0               -> overshooting: HALVE the lr
      3 stalls in a row       -> converged; report it so the caller can stop

    Doubling on a small-but-positive gain is deliberate. The usual instinct is to lower
    the rate when progress slows, but that is the right move only when the metric is
    getting *worse*; when it is merely crawling, a lower rate makes it crawl slower.
    """

    def __init__(self, opt, good=0.01, max_lr=1e-3, min_lr=1e-7, patience=3):
        self.opt, self.good = opt, good
        self.max_lr, self.min_lr, self.patience = max_lr, min_lr, patience
        self.prev = None
        self.stalls = 0

    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def _set(self, v: float) -> None:
        v = max(self.min_lr, min(self.max_lr, v))
        for g in self.opt.param_groups:
            g["lr"] = v

    def step(self, metric: float) -> str:
        """Returns a short verdict string for the log."""
        if self.prev is None:
            self.prev = metric
            return f"baseline {metric:.4f}"
        gain = metric - self.prev
        self.prev = metric
        cur = self.lr()
        if gain <= 0:
            self.stalls += 1
            self._set(cur / 2)
            return (f"gain {gain:+.4f} <= 0 -> lr {cur:.2e} -> {self.lr():.2e} "
                    f"(stall {self.stalls}/{self.patience})")
        if gain < self.good:
            self.stalls += 1
            self._set(cur * 2)
            return (f"gain {gain:+.4f} too slow -> lr {cur:.2e} -> {self.lr():.2e} "
                    f"(stall {self.stalls}/{self.patience})")
        self.stalls = 0
        return f"gain {gain:+.4f} healthy -> lr {cur:.2e} unchanged"

    @property
    def converged(self) -> bool:
        return self.stalls >= self.patience


def save_rolling(path: str, payload: dict) -> None:
    """Write one rolling checkpoint, replacing the previous one.

    Only the newest periodic checkpoint is kept, to stay inside the disk budget - but it
    is written to a temp file and then renamed, never overwritten in place. Overwriting
    the only checkpoint directly means a crash mid-write leaves a truncated file and the
    whole run is unrecoverable; os.replace is atomic, so either the old or the new file
    is present, never a half-written one.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _require_cuda():
    # Abort rather than silently train on CPU. Installing anything that depends on
    # torch (facexlib, diffusers, transformers) pulls a CPU build into the workspace
    # venv, and PYTHONPATH is searched before the interpreter's own site-packages, so
    # that CPU build shadows the CUDA one. Training then runs ~50x slower with no
    # error - just a "pin_memory ... no accelerator found" warning in the log. That
    # cost two runs before it was spotted, so it is a hard failure now.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "  REFUSING TO TRAIN ON CPU: torch " + torch.__version__ +
            " reports no CUDA. A CPU torch is probably shadowing the CUDA build on "
            "PYTHONPATH. Check: python -c \"import torch;print(torch.__file__)\"")


def main() -> None:
    _require_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["msrresnet", "msrgan", "esrgan", "realesrnet", "realesrgan"])
    ap.add_argument("--steps", type=int, default=20000)   # derived budget, see docstring
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)     # half the authors' scratch rate
    ap.add_argument("--gt-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=0)  # 0 = auto: every 5 epochs
    ap.add_argument("--save-every-epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <model>_last.pth instead of the published weights")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = build(args.model, device)
    n_p = sum(p.numel() for p in model.parameters())
    tr = DataLoader(PairSet("train", args.gt_size), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, persistent_workers=args.workers > 0,
                    drop_last=True, pin_memory=True)
    n_train = len(tr.dataset)
    spe = max(1, n_train // args.bs)
    print(f"  model {args.model}  {n_p/1e6:.2f} M params  from {os.path.basename(ckpt)}", flush=True)
    print(f"  N_train {n_train}  batch {args.bs}  {spe} steps/epoch  "
          f"-> {args.steps} steps = {args.steps/spe:.0f} epochs  lr {args.lr}", flush=True)

    before = evaluate(model, "val", device)
    print(f"  BEFORE fine-tune: val PSNR {before['psnr']:.3f}  SSIM {before['ssim']:.4f}", flush=True)
    save_inference(model, device, os.path.join(out_dir, "inference"), "before")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-7)

    # Check every 5 epochs, as instructed. steps/epoch = N_train // batch, so the
    # cadence follows the dataset rather than being a fixed step count.
    spe = max(1, n_train // args.bs)
    if args.val_every <= 0:
        args.val_every = max(50, spe * 5)
    tuner = LRAutoTune(opt, good=0.01, max_lr=1e-3)
    print(f"  autotune every {args.val_every} steps (= {args.val_every/spe:.0f} epochs)",
          flush=True)
    best = {"psnr": before["psnr"], "step": 0}
    hist = []
    step = 0
    last_path = os.path.join(out_dir, f"{args.model}_last.pth")

    # Resuming restores the optimiser and scheduler too, not just the weights. Adam's
    # moment estimates and the cosine schedule's position are part of the training
    # state; reloading weights alone silently restarts the schedule at its peak LR and
    # throws away the momentum, which looks like the model getting worse on resume.
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step = ck["step"]; best = ck["best"]; hist = ck.get("history", [])
        print(f"  RESUMED from step {step} (best val PSNR {best['psnr']:.3f} "
              f"at step {best['step']})", flush=True)

        # The restored scheduler carries its OWN base_lr and T_max, and it rewrites the
        # optimiser's lr on every .step(). So a --lr passed on a resume is silently
        # discarded: asking for 2e-4 kept training at the old 1e-4 schedule, by then
        # decayed to 5e-5, and the run sat flat for 6,000 steps looking converged.
        # If the caller asked for a different rate, honour it - rebuild the schedule
        # from the current step at the new base.
        restored = sched.base_lrs[0] if getattr(sched, "base_lrs", None) else None
        if restored and abs(restored - args.lr) / max(args.lr, 1e-12) > 0.01:
            for g in opt.param_groups:
                g["lr"] = args.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.steps - step), eta_min=1e-7)
            print(f"  LR OVERRIDE: restored schedule was base {restored:.2e}; "
                  f"caller asked {args.lr:.2e} -> rebuilt cosine over the remaining "
                  f"{args.steps - step} steps", flush=True)
    elif args.resume:
        print(f"  --resume given but {os.path.basename(last_path)} not found; "
              f"starting from the published weights", flush=True)

    save_epochs = max(1, args.save_every_epochs)
    save_every = save_epochs * spe          # 5 epochs expressed in steps
    t0 = time.perf_counter()
    model.train()
    while step < args.steps:
        for lr_b, hr_b in tr:
            if step >= args.steps:
                break
            lr_b, hr_b = lr_b.to(device, non_blocking=True), hr_b.to(device, non_blocking=True)
            out = model(lr_b)
            loss = F.l1_loss(out, hr_b)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % args.val_every == 0 or step == args.steps:
                v = evaluate(model, "val", device)
                hist.append({"step": step, "loss": float(loss.detach()), **v})
                flag = ""
                if v["psnr"] > best["psnr"]:
                    best = {"psnr": v["psnr"], "step": step}
                    torch.save({"params": model.state_dict()},
                               os.path.join(out_dir, f"{args.model}_best.pth"))
                    flag = "  *best"
                verdict = tuner.step(v["psnr"])
                print(f"  step {step:6d}/{args.steps}  loss {float(loss.detach()):.4f}  "
                      f"val PSNR {v['psnr']:.3f}  SSIM {v['ssim']:.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}{flag}", flush=True)
                print(f"      autotune: {verdict}", flush=True)
                # The cosine schedule is rebuilt from the tuner's rate so the decay
                # continues from the NEW lr rather than snapping back to the old base on
                # the next sched.step().
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, args.steps - step), eta_min=1e-7)
                if tuner.converged:
                    print(f"  CONVERGED: {tuner.patience} stalled checks in a row at "
                          f"step {step}. Stopping early; best checkpoint kept "
                          f"({best['psnr']:.3f} @ {best['step']}).", flush=True)
                    args.steps = step          # ends the loop cleanly
                model.train()

            # Rolling checkpoint every N epochs. One file, replaced each time, so disk
            # use stays flat however long the run goes. `_best.pth` is separate and is
            # never rotated - the newest checkpoint is not necessarily the best one.
            if step % save_every == 0:
                save_rolling(last_path, {
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "step": step, "best": best,
                    "history": hist, "args": vars(args),
                })
                print(f"  checkpoint saved at step {step} "
                      f"(epoch {step/spe:.0f}) -> {os.path.basename(last_path)}", flush=True)

    mins = (time.perf_counter() - t0) / 60
    # Final rolling save, so `--resume` can always continue from where this stopped even
    # if the run ended between periodic saves.
    save_rolling(last_path, {
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "step": step, "best": best,
        "history": hist, "args": vars(args),
    })
    bp = os.path.join(out_dir, f"{args.model}_best.pth")
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device)["params"], strict=True)
    res = {s: evaluate(model, s, device) for s in ("train", "val", "test")}
    save_inference(model, device, os.path.join(out_dir, "inference"), "after")

    print(f"\n  trained {args.steps} steps in {mins:.1f} min; best val PSNR "
          f"{best['psnr']:.3f} at step {best['step']}")
    print(f"  {'split':7s} {'PSNR':>8s} {'SSIM':>8s}")
    for s in ("train", "val", "test"):
        print(f"  {s:7s} {res[s]['psnr']:8.3f} {res[s]['ssim']:8.4f}")
    gain = res["val"]["psnr"] - before["psnr"]
    print(f"  BEFORE (val) {before['psnr']:.3f} -> AFTER (val) {res['val']['psnr']:.3f} "
          f"= {gain:+.3f} dB")
    if gain < 0.05:
        print(f"  NOT SATISFIED: val PSNR moved {gain:+.3f} dB. Continue training with:")
        print(f"    python train_sr.py --model {args.model} --resume "
              f"--steps {args.steps * 2} --lr {args.lr/2:.1e}")
        print(f"  (--resume restores optimiser and scheduler state from "
              f"{os.path.basename(last_path)}; halving the LR is the usual next step "
              f"when a fine-tune has flattened rather than diverged)")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "params_M": round(n_p/1e6, 3),
                   "n_train": n_train, "batch": args.bs, "steps": args.steps,
                   "epochs_equiv": round(args.steps/spe, 1), "lr": args.lr,
                   "gt_size": args.gt_size, "minutes": round(mins, 2),
                   "before_val": before, "final": res, "history": hist}, fh, indent=2)


if __name__ == "__main__":
    main()
