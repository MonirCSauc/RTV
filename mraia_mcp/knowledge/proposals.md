---
title: Proposed modifications (M1 to M5), open to challenge
summary: The five modifications from the proposal deck, each with the idea, reasoning, where it touches the code, risks and how to test it. Recommendations only; better alternatives are welcome.
---

# Proposed modifications M1 to M5

**Status: recommendations from the proposal (Nov 2026), not commitments.** They were written before any training
run. Treat each one as a hypothesis. If the code, a paper or an experiment shows a better route to the same goal,
say so, explain why, and record it with `suggest_modification`. The team decides.

Goals they serve: generalise across customers (loose clothing, busy store) and across garments (a catalog), with
little or no added inference cost.

---

## M1: body-part map rendered from SMPL (replace DensePose)

- **Idea.** Draw the body-part map from BEV's SMPL body (already computed each frame) instead of DensePose, in both
  training and the mirror. Loose clothes then no longer corrupt the map, and DensePose leaves the pipeline (faster).
- **Why.** The network must see at training time the same kind of input it gets at run time. RTV fixes loose
  clothing only for the recording model (BodyMap) and assumes tight clothing at inference (paper section 5).
- **Code it touches.** Body map creation in `VITON/viton_upperbody.py` / `VITON/viton_fullbody_seq.py`, dataset
  generation (`DatasetGeneration/upperbody_dataset_generation.py`), the renderer (`OffscreenRenderer/`, SMPL templates
  with a per-part colour texture), the dataset loader.
- **Main risk.** BEV's fit is not pixel exact, so the map may not line up with the visible body edge. That is why M2
  exists.
- **Test.** Body-part IoU against the same person in tight clothing (DensePose as reference) or BEDLAM ground truth;
  FPS gain; image quality on held-out frames.

## M2: refinement network (optional)

- **Idea.** A small network, trained on many people plus synthetic bodies (BEDLAM, which has loose clothing with the
  true body known), that aligns the M1 map to the visible body. Only if M1 is not accurate enough.
- **Limit.** Under loose clothes the true body shape is unobservable; a general model can only guess average
  proportions.
- **Cost.** Needs the cloud GPU; BEDLAM is huge (BEDLAM2.0 is about 26 TB), so use a subset.

## M3: customer isolation

- **Idea.** A mirror zone (only people inside it count), tracking across frames, and one shared person choice for BEV
  and DensePose so bystanders cannot take the garment.
- **Why.** Upstream BEV keeps the nearest person and DensePose the largest box; nothing checks they agree. (Our fork
  already added person matching; see `fork_changes`.)
- **Rejected sub-idea.** Face recognition per frame: face hidden when turning, extra cost, biometric privacy. Possible
  only as a one-time session login.
- **Test.** Failure rate and garment switches per minute with bystanders.

## M4: baseline fixes (engineering)

- ConvLSTM memory for the upper-body path (needs a clip dataset class and a training script; the model side exists).
- Crop smoothing for the upper-body path.
- Soft alpha compositing, hands and head kept in front, hold last good garment on missed frames.
- GPU speed-ups: warps and compositing on the GPU, frame-grabber thread, fp16 / TensorRT, parallel stages.
- Fix the label resize (`interpolation=cv2.INTER_NEAREST`) and the VGG offset (`+0.5`).
- Several of these are already done in our fork (`fork_changes`).

## M5: shared backbone + per-garment parts

- **Idea.** Shared encoder + the 9 residual blocks (about 93 percent of the 182M weights) learn bodies and poses from
  every garment and recorded body. A small garment code steers the shared blocks (FiLM: $h' = \gamma_k \odot h +
  \beta_k$). A small per-garment decoder (about 6 to 9M weights, 25 to 35 MB instead of 730 MB) paints the garment.
- **Training.** Stage 1: shared part on many garments at once. Stage 2: per new garment, train only its decoder and
  code (transfer learning). Fallback ladder: small adapters, adapters + own decoder, full network initialised from
  the shared one.
- **Why.** One 730 MB network per garment trained on one body does not scale to a catalog or to varied customers.
- **Risk.** Fidelity may drop compared with a dedicated network; needs multi-garment data and the cloud GPU.
- **Test.** Held-out frames of each recording: PSNR, SSIM, LPIPS, FID against the dedicated network; training time,
  frames and MB per new garment; unseen body shapes.

---

## Open questions worth challenging

- Is M1's SMPL map accurate enough without M2, or would a light keypoint heatmap (as in the paper's Sapiens input)
  be a better garment-invariant signal?
- Would fine-tuning a dedicated network from an existing checkpoint (`--load_pretrain`) get most of M5's training-time
  gain with less risk?
- Is pix2pixHD still the right generator, or would a lighter modern architecture give better fidelity per FPS?
- Should the upper-body ([57]) path or the full-body recurrent path be the main target?
