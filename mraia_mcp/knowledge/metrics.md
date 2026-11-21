---
title: Evaluation plan and metrics
summary: What we measure for speed, stability, fidelity, body accuracy and scaling, with each metric's meaning and direction, and the robustness test set.
---

# Evaluation plan and metrics

## What each metric means

| Metric | Measures | Better |
|---|---|---|
| PSNR | Pixel error against the real frame | higher |
| SSIM | Structural match (luminance, contrast, structure) to the real frame | higher |
| LPIPS | Perceptual distance in deep-network feature space | lower |
| FID | Distance between feature distributions of generated and real images (Inception) | lower |
| KID | Like FID but unbiased for small sample sizes (good for about 3,000 frames) | lower |
| VFID | FID on video features (I3D), includes motion and smoothness | lower |
| FVD | Fréchet Video Distance, similar to VFID | lower |
| Warping error | Flicker beyond the person's real motion: warp frame t-1 to t with optical flow, measure the difference | lower |
| Body-part IoU | Overlap of predicted body-part map with reference map | higher |
| FPS / latency | Frames per second; camera-to-display delay per stage (median and 95th percentile) | higher / lower |

Wu et al. report KID, FID and VFID between the try-on video and a real video of the same person wearing the garment
(similar but unsynchronised motion), and FPS on an RTX 4090 (10.50 for the recurrent model, 12.38 for [57]).
The released code has no metric or timing code; our fork added latency logging.

## Evaluation plan (from the proposal)

| Goal | Metrics | Tests |
|---|---|---|
| Speed | FPS; camera-to-display latency per stage on the RTX 4080 | All; removing DensePose should raise FPS |
| Stability | Warping error; frame-difference maps | Memory fixes; no added flicker from new methods |
| Fidelity and realism | PSNR, SSIM, LPIPS on held-out frames; KID, FID, VFID against real videos | Shared backbone vs dedicated networks; all vs Wu et al. |
| Body under loose clothing | Body-part IoU against the same person in tight clothing, or BEDLAM ground truth | Body-map methods |
| Store and scaling | Failure rate; garment switches per minute with bystanders; training time, frames and MB per new garment | Isolation; shared backbone |

Final check: a user study (preference against Wu et al., realism ratings 1 to 5). No reviewed paper reports
camera-to-display latency, so measuring it is a contribution in itself.

## Test data

- Held-out chunk: the last part of each garment recording, never trained on.
- Robustness set: tight vs loose clothing x plain vs cluttered background, plus bystanders and two lighting setups.
- Run every method offline on the same test videos.
- Libraries: `torchmetrics` or `piq` for SSIM/LPIPS/FID/KID; keep evaluation code separate and unchanged while
  comparing methods.
