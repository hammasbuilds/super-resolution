# Super Resolution

Single-image super-resolution at 4x upscaling. Several architectures are fine-tuned on
the same dataset with the same budget and evaluated under one protocol, so the numbers
sit in a single comparable table.

*A demonstration / portfolio project.*

---

## Status

Results are being regenerated on **DIV2K**, the standard super-resolution benchmark
(~2040x1356 images).

The previous results were trained on 400x600 source photographs. At 4x that gives a
100x150 input, which is below the point where detail can be reconstructed rather than
invented - and it showed: four architectures spanning 1.52 M to 16.7 M parameters all
converged within 0.12 dB of one another. That was the dataset's information ceiling,
not a finding about the models, so those numbers have been withdrawn rather than left
up.

DIV2K gives a ~510x339 input at the same scale factor - eleven times the pixels - and is
what ESRGAN, Real-ESRGAN, SRGAN and EDSR are all trained on, so the results will be
directly comparable to published figures.

---

## Method

| Stage | Purpose |
|---|---|
| `baseline_classical.py` | nearest / bilinear / bicubic / Lanczos - the bar a learned model must clear |
| `eval_pretrained.py` | published checkpoints, zero-shot, no training |
| `train_sr.py` | fine-tuning, with learning-rate auto-tuning every 5 epochs |
| `build_div2k.py` | dataset construction: 480x480 crops at native pixel scale |
| `make_showcase.py` | inference panels: input -> output -> original |

Classical interpolation is evaluated alongside every model because a learned model is
only worth its weights if it beats what `Image.resize` gives for free.

Metrics are PSNR and SSIM on the Y channel with a 4-pixel border crop, the standard
protocol in the literature.

---

## Training

Hyperparameters are derived from the dataset rather than copied from reference
configurations written for different dataset sizes and GPU counts:

```
steps = epochs x N_train / batch
```

Batch size is chosen by measuring throughput. The learning rate is adjusted
automatically during training - checked every 5 epochs, raised when progress is real but
slow, lowered when the metric falls, and training stops once the metric stops beating
its best.

---

## Usage

```bash
python build_div2k.py          # build the dataset from DIV2K
python baseline_classical.py   # classical baselines
python eval_pretrained.py      # published checkpoints, no training
python train_sr.py --model msrresnet --steps 20000 --bs 32 --lr 1e-4
python make_showcase.py        # inference panels
```

Training resumes with `--resume`, restoring optimiser and scheduler state alongside the
weights.

---

## Acknowledgements

Architectures and pretrained weights from [BasicSR](https://github.com/XPixelGroup/BasicSR)
and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.
Dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (ETH Zurich).
