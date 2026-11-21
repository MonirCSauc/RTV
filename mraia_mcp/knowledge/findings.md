---
title: Findings log
summary: What we have learned from the code, papers and experiments, including why the output looks like a filter and the fixes sorted by effort. New findings are appended at the end.
---

# Findings

## From the repo walk-through (Nov 2026)

- **BodyMap vs per-garment network.** BodyMap is one per person (the recording model), maps VM + Sapiens heatmaps to
  a clean body map, and is used only while building the garment dataset. The per-garment network (ReGarSyn) maps
  VM + body map to garment + mask and runs in the mirror. BodyMap is not used at run time: it knows one person,
  DensePose is more exact on tight clothes, and Sapiens + BodyMap would cost FPS.
- The BodyMap training video and the garment videos must use the same person with similar movements.
- The repo's dataset script is the old method only (VM from BEV, SDP from DensePose, no BodyMap). Using it on loose
  garments reproduces the broken-map mismatch.
- RTV's full method is only partly reproducible from public code: `demo.py` only runs two released full-body
  checkpoints. Rebuilding the training pipeline is a necessary first step.
- Training on systematically broken inputs does not make clean inputs easier: the network learns shortcuts from the
  errors (like errors-in-variables bias in regression). Train on the kind of input the mirror will see.

## Why the output looks like a "filter" (compositing)

- Hard 0/1 mask (`alpha > 128`) eroded by 1 px; the mask edge flips frame to frame (shimmer).
- Studio lighting baked into the garment, no adaptation to the store.
- The customer's own clothes are not removed and peek out.
- Nothing keeps hands in front of the garment; stretched arms can leave the crop.
- Dark rim from black-background JPEG training garments (why the erode exists).
- Warps and compositing run on the CPU at full frame size.

## Fixes sorted by effort

**Quick wins:** soft alpha in compositing; train on straight colour (fill edge pixels with garment colour), save
garments as PNG, add a mask loss; hands and head back on top; smooth alpha over time; crop smoothing for the upper
body; hold the last good garment for a few frames when BEV or DensePose fails; warps and compositing on the GPU;
separate frame-grabber thread; fp16 / TensorRT; parallel stages; fix the label resize and the VGG offset.

**Moderate:** recurrent upper-body model (clip loader + training script); lighting harmonisation (match garment
brightness and colour to the customer region, statistics first); mirror zone + tracking + one shared person choice;
fine-tune from an existing checkpoint; a recording protocol with rare poses.

**Research:** clothing-robust body input at run time; shared backbone with per-garment parts; removing the
customer's own clothes (inpainting), a stretch goal.

## Log

<!-- record_finding appends dated entries below this line -->
