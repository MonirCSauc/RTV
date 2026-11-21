---
title: Where our fork departs from upstream
summary: Files our fork changed or added and the main behaviour changes (Windows support, person matching, soft GPU compositing, fixes, latency logging).
---

# Where our fork departs from upstream

Check the current state with `git diff upstream/main` (add the remote with `git remote add upstream https://github.com/ZaiqiangWu/RTV.git` and `git fetch upstream`). This summary was written on 2026-11-01 and may be out of date.

For reference only; the lessons above teach upstream. Read with `git diff upstream/main` in the fork. Files changed
(ignoring line endings): dataset generation and loader, offline inference and its options, most of
`OffscreenRenderer/`, `README.md`, `SMPL/my_bev.py`, `smpl_np.py`, `smpl_regressor.py`, the training script,
both `VITON/` pipelines, `pix2pixHD_rgba.py`, `pix2pixHD_rnn_rgba.py`, `requirements.txt`, `rtl_demo.py`,
`training_instructions.md`, `util/cv2_trans_util.py`, `util/densepose_util.py`. Files added: `composition/compositor.py`,
`composition/gpu_ops.py`, `model/garment_model.py`, `util/debug_view.py`, `util/gpu_check.py`, `util/latency.py`,
`util/matte_util.py`, `profile_demos.py`.

Main behaviour changes in the fork: a separate frame-grabber thread; Windows support (GLFW renderer) and more
robust camera opening; person matching between BEV and DensePose; hold on missed frames and person-change detection;
crop smoothing in the upper-body path; a `--body_shape` option; soft compositing on the GPU with hands and head kept
in front; straight-colour training targets (`fill_from_interior`), garments saved as PNG, and `rtv_meta.json` to tell
the compositor which colour convention a checkpoint uses; the VGG offset fixed to `+0.5`; the label resize fixed to
nearest neighbour; RAFT loaded only for training; latency instrumentation (`--latency_csv`) and the four CSV files in
the fork's root.
