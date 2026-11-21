---
title: Related work and datasets
summary: The eight reviewed try-on methods with speed, loose-clothing support and reported metrics; datasets the field uses; the three research gaps.
---

# Related work and datasets

## Reviewed methods

| Method | Input | Speed | Loose clothing | Main limitation | Reported quality |
|---|---|---|---|---|---|
| VITON (Han et al., 2018) | Image | not reported | No | 256 x 192 | IS 2.514; user study 77.2% |
| VITON-HD (Choi et al., 2021) | Image | not reported | No | tops, one frame | SSIM 0.895, LPIPS 0.053, FID 11.74 |
| ViViD (Fang et al., 2024) | Video | 1.44 FPS | No | slow, indoors | SSIM 0.949, LPIPS 0.068, VFID 3.405 |
| Fashion-VDM (Karras et al., 2024) | Video | 8 min / 64 frames | No | minutes per video | FID, FVD, CLIP |
| MagicTryOn (Li et al., 2025) | Video | 1.97 FPS | No | whole clip, 14B model | VFID 8.40, SSIM 0.901, LPIPS 0.060 |
| **Wu et al. (2025), our base** | Stream | 10.5 FPS (4090) | Tight only at inference | 12 h per garment | KID, FID, VFID; FID 42.88 (hanfu) |
| TryOnCrafter (Sun et al., 2026) | Video | about 1,360 s / video | No | not real time | VFID 9.61, SSIM 0.868, LPIPS 0.057 |
| LiveVVT (Cao et al., 2026, arXiv 2608.26714) | Stream | 22.39 FPS, 1.56 s first chunk | No | about 20 GB peak memory | VFID 13.19, SSIM 0.838, LPIPS 0.099 |

Other relevant work: SwiftTry (2024, fast video try-on, trained on one A100), Champ (2024, maps drawn from SMPL for
animation), URAvatar and MegaPortraits (shared backbone pattern for avatars), TripVVT (2026, multi-person video try-on
inside a diffusion model, no speed reported), BEDLAM (synthetic bodies with ground truth).

## Datasets

- VITON-HD: 13,679 frontal person/garment pairs, 1024 x 768, women's tops.
- Dress Code: image try-on, used with VITON-HD.
- ViViD: 9,700 garment-video pairs, about 1.21M frames at 832 x 624, mostly indoors.
- Per-garment recordings (Wu et al.): about 3,000 images per garment, SAM2 masks.
- BEDLAM / BEDLAM2.0: synthetic bodies in loose clothing with the true SMPL body known (BEDLAM2.0 about 26 TB).
- None contains customers in their own loose clothing or busy store scenes, so we record our own.

## The three gaps

1. The customer's own clothing: every method reads the body from parsing, DensePose or a silhouette of the worn
   clothes; none evaluates a customer in a coat, hoodie or wide dress.
2. The store environment: no reviewed work measures clutter, uneven lighting or other shoppers.
3. One network per garment: Wu et al. train a full network per garment (about 12 h, 730 MB) on one person's body.

Speed is a constraint, not a gap: any fix must add little or no inference cost. Novelty check (Nov 2026): the
loose-clothing-at-inference direction and the shared backbone for try-on appear open; Wu's 2025 PhD dissertation was
not checked yet.
