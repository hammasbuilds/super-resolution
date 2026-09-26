# MSRGAN

16-block SRResNet generator, 1.52 M parameters, fine-tuned from the BasicSR MSRGAN
published checkpoint.

| Split | PSNR (dB) | SSIM |
|---|---:|---:|
| Val   | 28.20 | 0.8761 |
| Test  | 28.50 | 0.8923 |

Bicubic reference on the same test set: 26.38 dB. **+2.12 dB.**

Starting from a GAN-trained checkpoint, this model gained 1.83 dB over its published
zero-shot score of 26.67 dB — the largest improvement of any model in the set, since a
GAN-trained network begins further from the pixel-accuracy optimum than an L1-trained
one.

Training stopped automatically once validation PSNR had been flat for three consecutive
checks. Full history in `training_results.json`.
