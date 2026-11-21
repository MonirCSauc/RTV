---
title: Paper vs code audit
summary: What the released code implements as in the paper, differently, what is missing (BodyMap, recurrent training, metrics) and what is extra or dead code.
---

# Paper vs code audit

## 1.4 Implemented as in the paper, differently, missing, extra

### As in the paper
- BEV-based SMPL fitting and the grid-textured VM render, including the full-body template the paper uses.
- The simplified body map without UV and with merged torso (`IUV2SSDP`), used by the full-body pipeline.
- The ReGarSyn architecture (pix2pixHD with a ConvLSTM inserted in the decoder) and its recurrent inference.
- pix2pixHD's per-frame losses and the optimiser settings (Adam, $\text{lr}=2\times10^{-4}$, $\beta_1=0.5$).
- Inference from DensePose assuming tight clothing, and composition with the predicted mask.

### Implemented differently
- **Two methods in one repository.** The upper-body path (`rtl_demo.py`, dataset generation, training, offline
  inference) is the earlier method [57]: upper-body VM + DensePose SDP *with* UV, one image at a time, 512×512 crops.
  Only the full-body inference path (`demo.py`) is the paper's method.
- **$\beta$ fixed to zero** zeroes only betas 0 to 3 and 10 (BEV predicts 11: 10 SMPL shape values plus an age
  offset), not all. The live upper-body pipeline does not fix the shape at all, although its training data did:
  a train/inference mismatch.
- **Training loss**: the recurrent model adds an optical-flow (RAFT) temporal loss and an L1 mask loss, against the
  paper's statement of no temporal loss. The VGG loss is on unmasked colour for the recurrent model but on masked
  colour for the upper-body model. The upper-body model (`pix2pixHD_rgba.py`) has **no mask loss at all**: its alpha
  channel is learned only through the GAN, feature-matching and masked-VGG terms.
- **Loss weight**: `--lambda_feat` defaults to 10 and `training_instructions.md` does not override it, whereas [57]
  describes weights of 1 for the feature-matching and VGG terms.
- **Epochs**: `training_instructions.md` trains 80 + 80 (decay) epochs at batch size 4, against 40 epochs in this
  paper and 100 epochs at batch size 8 in [57].
- **Alpha value range** differs between the two model families. Upper body: the dataset does *not* rescale the mask,
  so the target alpha is in $[0,1]$ and inference does `clamp(0, 1)` on the generator's tanh channel. Recurrent: the
  full-body processor maps the tanh output with $\alpha = \text{clamp}(a,-1,1)/2 + 0.5$, which implies a target in
  $[-1,1]$. No script in the repository shows how the recurrent targets were built, so treat this as inferred.
- **Compositing** thresholds the mask (soft edges are lost) and erodes it by one pixel; the paper's Eq. 5 allows a
  soft mask.
- **Where the ConvLSTM sits**: after three of the four decoder up-sampling steps (half the input resolution, 128
  channels), not at the bottleneck. A comment in `RNNGenerator.inference_forward` ("36x27") is left over from an
  earlier design and is wrong for the current code. See 1.6 for how this was verified.
- **Gradient through time**: only the last 8 frames of a clip get gradients; earlier frames warm the state without
  gradients (truncated back-propagation through time). The paper only says clips of 8 to 60 frames.
- **Body semantic map at training time** comes from DensePose (upper-body generation), not from BodyMap.
- **Crop window.** Upper body (`SMPL_Regressor.get_trans2roi`): a square centred at $0.8\,J_9 + 0.2\,J_{12}$ (spine
  and neck joints, i.e. the upper chest) with half-side $1.45\,\lVert J_{15}-J_0\rVert$ (head to pelvis). Full body
  (`get_fullbody_trans2roi`): a 576×432 window centred on the pelvis $J_0$, height $1.4\times$ the head-to-feet distance.
  Upper-body training crops are cut at 1024 × 1024 and resized to 512; the live demo cuts 512 directly with the same
  framing.
- **Crop smoothing over time** exists only in the full-body path (`TemporalSmoothing`, $c=0.9$). The upper-body
  path recomputes the crop from scratch every frame, which adds jitter.
- **Person selection.** BEV keeps the person nearest the camera (smallest depth); DensePose keeps the largest
  bounding box. Nothing checks that they are the same person.

### Missing
- Sapiens joint heatmaps, the garment-invariant representation, and the BodyMap network (the paper's first
  contribution).
- Full-body / loose-garment dataset generation (§3.4) and a training script and sequence dataset for ReGarSyn.
- Evaluation code for KID, FID and VFID, the ablation variants, and any timing code.
- Garment removal (also listed as a limitation in the paper).

### In the repository but not in the paper
- The upper-body [57] pipeline, Qt demo app and offline video script.
- 9 unused pix2pixHD variant files and 15 `util/` modules that nothing imports (`binary_util`, `diff_tex_sampling`,
  `error_measure`, `gdrive_downloader`, `get_laplacian_matrix`, `html2`, `human_pose`, `image_quantization`,
  `image_writer`, `large_video_loader`, `mask2roi`, `optical_flow_mask_refine`, `random_affine_batch`,
  `skeleton_annotator`, `temporal_variance_heatmap`).
- Vendored DensePose/detectron2 project code; RAFT and VGG19 weights from torchvision.

## 1.5 Where we would change things for our goals

| Goal | Most likely places | Why |
|---|---|---|
| **Less jitter and inconsistency** | Use the recurrent model for upper body too (`pix2pixHD_rnn_rgba.py`, `RNNGenerator`) and write its training script; add crop smoothing to the upper-body path (`util/cv2_trans_util.py:TemporalSmoothing`); pose filtering (`SMPL/pose_filter.py`, unused); a soft mask in compositing | The upper-body networks are per-frame, so they flicker by design (§1); the upper-body crop is recomputed every frame; the thresholded mask makes edges pop on and off. |
| **Higher fps, lower latency** | DensePose (render the body map from the SMPL fit instead, or a smaller input); fp16 / TensorRT for the generator; run BEV and DensePose concurrently, and capture frames in their own thread (`VITON/viton_upperbody.py`, `rtl_demo.py`); move warps and compositing to the GPU | Everything runs one stage after another in one thread, and warps and compositing run on the CPU at full frame size. The paper reports about 12 fps for [57] on an RTX 4090 (Table 2). Upstream has no timing code, so we must add our own before optimising. |
| **Our own garments** | `DatasetGeneration/upperbody_dataset_generation.py`, `Datasets/upperbody_garment`, `Training/upperbody_training.py`, `training_instructions.md`; fine-tuning from an existing checkpoint (`--load_pretrain`) | This pipeline already trains new upper-body garments from a phone video + SAM masks; fine-tuning could cut the long per-garment training. |
| **Outfit recommendation** | New module, outside the try-on pipeline, plus garment metadata next to each checkpoint and the garment list in `rtl_demo.py` | Nothing in the paper or code covers it; it needs garment attributes or image embeddings and a compatibility model or rules. |
| **Lower-body garments (pants, shorts)** | A lower-body VM template (like `SMPL/upperbody_smpl`), a lower-body crop window (`SMPL_Regressor`), dataset generation and training variants, and a second network composited under/over the upper garment (`VITON/`, `composition/`) | Only upper-body and full-body (dress/coat) templates and pipelines exist; pants need leg parts, a different crop, and layering with the top. |
| **Robust to background noise** | Person selection in BEV (depth order) and DensePose (largest box) made consistent; a customer tracker / mirror zone; person segmentation before compositing | Bystanders and clutter corrupt BEV and DensePose, and the two can lock onto different people. |
