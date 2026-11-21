---
title: How the RTV pipeline works
summary: End-to-end data flow from camera frame to mirror, the two methods in one repo, and the conventions (colour order, value ranges, sizes, affine matrices).
---

# How the RTV pipeline works

## Two methods live in one repository

- **Upper body, per frame ("[57]", the earlier method, arXiv 2506.10468).** `rtl_demo.py`, dataset generation,
  training and offline inference. Input = upper-body VM render + DensePose SDP map (with UV), 512 x 512 crops, one
  frame at a time (no memory, so it flickers by design).
- **Full body, recurrent (the 2025 paper).** `demo.py` only, inference with two released checkpoints. Input = full-body
  VM + simplified DensePose map (SSDP, no UV, merged torso), 576 x 432 window, pix2pixHD with a ConvLSTM memory.
  No training script, no dataset script and no BodyMap code are released for this path.

## Per-frame data flow (inference)

1. **Camera frame.** Rotated 90 degrees clockwise, mirrored, resized to 1024 high, cropped to 9:16, so a 16:9 webcam
   gives 576 x 1024 (w x h). BGR (OpenCV).
2. **BEV body fit.** `SMPL/my_bev.py:MyBEV`, `SMPL/smpl_regressor.py:SMPL_Regressor`. Monocular multi-person SMPL
   fit; the person nearest the camera (smallest depth) is kept.
3. **Crop window.** Upper body (`get_trans2roi`): square centred at $0.8 J_9 + 0.2 J_{12}$ with half-side
   $1.45 \lVert J_{15} - J_0 \rVert$. Full body (`get_fullbody_trans2roi`): 576 x 432 centred on the pelvis, height 1.4 x
   head-to-feet, smoothed over time (`TemporalSmoothing`, c = 0.9). The upper-body path recomputes the crop every
   frame (jitter).
4. **VM render** ($I_{vm}$, "virtual measurement garment"). SMPL mesh without head, hands and feet, textured with a
   6-colour noise pattern, rendered off-screen (`OffscreenRenderer/`, `SMPL/upperbody_smpl`, `SMPL/fullbody_smpl`).
   Shows 3D pose and orientation.
5. **DensePose IUV** (`model/DensePose/`). Converted to SDP (`util/densepose_util.py:IUV2SDP`, keeps UV, upper body)
   or SSDP (`IUV2SSDP`, part colours only, torso merged, full body). DensePose keeps the largest box: nothing checks
   it is the same person BEV chose.
6. **Generator.** VM and body map stacked (6 channels) into a pix2pixHD generator that outputs RGBA: garment colour +
   alpha mask. Recurrent variant keeps ConvLSTM state between frames.
7. **Compositing.** `composition/naive_overlay.py:naive_overlay_alpha`: alpha thresholded at 128, eroded 3 x 3, then
   $I_o = I \odot (1 - \tilde I_m) + \tilde I_g \odot \tilde I_m$ with a hard 0/1 mask. Warps and compositing run on
   the CPU at full frame size.

## Conventions that silently break things

- **Colour order.** Camera = BGR. VM render = RGB. Dataset generation saves the VM with `cv2.imwrite` (treats it as
  BGR) and the loader reads with PIL, so the network was trained on channel-reversed VM; inference therefore does
  `vm_tensor[:, [2, 1, 0]]`. The body map is never swapped (I, U, V fed as R, G, B). Network output is RGB;
  `tensor2im(..., rgb=False)` turns it into BGR for compositing.
- **Value ranges.** Network inputs and RGB outputs in $[-1, 1]$ (`x / 255 * 2 - 1`). Alpha: upper-body target in
  $[0, 1]$ (inference clamps); recurrent maps tanh with $\alpha = \mathrm{clamp}(a, -1, 1)/2 + 0.5$ (target inferred
  to be in $[-1, 1]$).
- **Sizes.** Upper-body networks 512 x 512; full-body 576 x 432 (h x w). Training crops cut at 1024 and resized to 512.
- **Affine matrices.** 2 x 3 OpenCV matrices map camera pixels to crop pixels (`trans2roi`); the inverse maps back.
  Change the crop and you change both.

## Why loose clothing breaks it

DensePose reads the body from the clothes' outline. Under loose garments the map is wrong. The paper fixes this only
for the recording model in training (BodyMap network, trained per person on a tight-clothing video) and still assumes
the customer wears tight clothing at inference. That is the gap M1 and M2 target.
