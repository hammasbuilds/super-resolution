# MSRResNet

16-block SRResNet, 1.52 M parameters, fine-tuned from the BasicSR published checkpoint.

| Split | PSNR (dB) | SSIM |
|---|---:|---:|
| Train | 28.62 | 0.8857 |
| Val   | 28.27 | 0.8763 |
| Test  | 28.55 | 0.8945 |

Bicubic reference on the same test set: 26.38 dB. **+2.17 dB.**

Trained 20,000 steps at batch 32, lr 1e-4 with cosine decay. Full history and the exact
arguments are in `training_results.json`.
