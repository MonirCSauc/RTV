---
title: Compute, storage and recording resources
summary: Current GPU, the pending cloud A100 request, rough GPU-hour estimates, the storage plan, and the recording setup.
---

# Compute, storage and recording

## Now

- One NVIDIA RTX 4080, 16 GB, Windows, AUC ML lab. Use it for code work, preprocessing (BEV, DensePose, frame
  extraction), debugging, short runs and per-garment upper-body training.

## Requested (pending)

- Cloud GPU subscription, NVIDIA A100 80 GB class, December 2026 to May 2027 (request sent through Dr. Noha, Nov 2026).
  Until it arrives, large runs (shared backbone, recurrent clips, refinement network) are paused.
- Internal estimate (not in the request): about 800 A100 hours (roughly 650 to 850): about 100 h per-garment models,
  100 h recurrent training, 300 to 450 h for 2 to 3 shared-backbone runs, 150 to 200 h ablations, evaluation and
  failed runs. Based on an unverified 12 h per garment on a 4090; measure with a one-epoch pilot first.
- A100 over H100: at this model's small batch size an H100 is estimated 1.3 to 1.7x faster but costs 1.6 to 2.2x more
  per hour.

## Storage plan for cloud runs

- Keep raw 4K videos and the master dataset copy on the local PC.
- Preprocess locally; upload only finished training data (512 crops + VM + body maps, roughly 1 to 2 GB per garment).
- Use a small persistent network volume (about 100 GB) next to the GPU; download only the best checkpoints (730 MB
  each per-garment model).

## Recording setup

- A phone with the free Blackmagic Camera app is enough; lighting matters more than the camera (shutter 1/250 or
  faster to avoid motion blur). Plain grey or white backdrop, two or three LED lights with softboxes, a tripod.
- Follow Wu et al.'s pose guide (`assets/pose_guidance/symmetric.pdf`), plus rare poses (crossed arms, arms raised,
  turning around). Same person and similar movements for related recordings.
