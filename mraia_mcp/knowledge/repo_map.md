---
title: Repository map (upstream RTV)
summary: Every folder and entry point, the external weights the code expects, and where each paper component lives in the code (or is missing).
---

# Repository map (upstream RTV, commit dd165d5)

Paths are relative to the repo root. Our fork changes some of these files; see `fork_changes`.

## 1.2 What is in the repository

| Folder / file | What it is | Used by |
|---|---|---|
| `rtl_demo.py` | Live upper-body try-on app (PyQt5 full-screen window, one capture-and-process thread, garment list) | entry point |
| `demo.py` | Live full-body try-on with the recurrent network (`coat_seq_vmssdp2ta_576`), no garment list | entry point |
| `Inference/upperbody_inference.py` | Offline: video file in, `./output.mp4` out, using a checkpoint from `./checkpoints/` | entry point |
| `DatasetGeneration/upperbody_dataset_generation.py` | Builds a per-garment upper-body dataset from a video + SAM masks | entry point (training) |
| `Training/upperbody_training.py` (+ `.sh`) | Trains one upper-body garment network | entry point (training) |
| `VITON/viton_upperbody.py` | `FrameProcessor`: per-frame upper-body pipeline and garment switching | live + offline |
| `VITON/viton_fullbody_seq.py` | `FullBodySeqFrameProcessor`: full-body recurrent pipeline | `demo.py` |
| `VITON/viton_fullbody.py` | Older full-body variant; cannot be imported (`from .ckpt_dict import ckpt_dict`, file missing) | dead |
| `SMPL/` | BEV wrapper (`my_bev.py`), `SMPL_Regressor` (pose, shape, crop transforms), numpy SMPL (`smpl_np*.py`, `model.pkl`), upper- and full-body VM templates, pose and translation filters | all pipelines |
| `OffscreenRenderer/` | OpenGL off-screen renderer for the VM image. Forces `PYOPENGL_PLATFORM=egl`, so it runs on Linux only | all pipelines |
| `model/DensePose/` | Vendored DensePose (detectron2 project) and `densepose_extractor.py:DensePoseExtractor` | all pipelines |
| `model/pix2pixHD/` | pix2pixHD networks and model wrappers; `pix2pixHD_rgba.py` (upper body), `pix2pixHD_rnn_rgba.py` (recurrent); 9 variant files that only `models.py` imports (ASAPNet x2, dual, mask, align, inpaint, utransformer, ui, plus `attention.py`), used by no demo or training script | all pipelines |
| `model/ConvLSTM/` | ConvLSTM cell and stack | recurrent model |
| `composition/naive_overlay.py` | Compositing (`naive_overlay_alpha`, hard mask) | inference |
| `Datasets/upperbody_garment/` | PyTorch dataset with random-affine augmentation | training |
| `OpticalFlow/` | RAFT (torchvision) optical-flow wrapper | recurrent model (training loss, but also loaded at inference) |
| `options/` | pix2pixHD-style command-line options (base / train / test) | training + model creation |
| `util/` | Image warps, video I/O, DensePose helpers, and 15 modules nothing imports (see 1.4) | everywhere |
| `assets/` | SMPL templates and sparse vertex maps (`assets/smpl`), the grid texture, garment thumbnails, pose guides | everywhere |

### External models and weights the code expects:
- **Garment checkpoints**: `rtv_ckpts` from Hugging Face (`wuzaiqiang/rtv_ckpts`, cloned with git-lfs as the README
  says). Upper body: `lab_03/04/07_vmsdp2ta`, `jin_17/18/22_vmsdp2ta` (pix2pixHD, 6 to 4 channels, 512×512, about 730 MB
  each). Full body: `coat_seq_vmssdp2ta_576` and `han_seq_vmssdp2ta_576` (recurrent, 576×432, about 739 MB each). Each
  folder holds one file, `latest_net_G.pth`. The name encodes the method: `vm` + `sdp` (or `ssdp`) inputs, `2ta`
  "to texture and alpha", `seq` recurrent, `576` the crop height.
- **BEV / ROMP** (`simple_romp`, installed from the authors' ROMP fork): monocular multi-person SMPL fitting;
  downloads its own weights.
- **DensePose R50-FPN s1x** (detectron2): weights downloaded from `dl.fbaipublicfiles.com` on first run.
- **SMPL body model**: `SMPL/model.pkl` and the meshes in `assets/smpl`.
- **RAFT-small** (torchvision weights): needed by the recurrent training loss, but `Pix2PixHD_RNN_RGBA.initialize`
  creates it unconditionally, so `demo.py` downloads and loads it too.
- **VGG19** (torchvision weights): pix2pixHD's perceptual loss, training only.
- Libraries (`requirements.txt`): Python 3.12, PyTorch 2.4, detectron2, OpenCV, PyQt5, PyOpenGL, trimesh, PyGLM,
  scipy. The README targets Ubuntu (apt packages) and an RTX 3060 or better.

## 1.3 Paper component → code

| Paper component (section) | Where in the repository | Status |
|---|---|---|
| BEV SMPL fit (§3.2) | `SMPL/smpl_regressor.py:SMPL_Regressor`, `SMPL/my_bev.py:MyBEV` | implemented |
| $\beta$ fixed to zero (§3.2) | `MyBEV.process_normal_image` with `fix_body=True` zeroes betas 0 to 3 and 10 only. Used by dataset generation and `viton_fullbody_seq.py`; the live upper-body `FrameProcessor` uses the default `fix_body=False` (the customer's own estimated shape) | differs (partial) |
| VM garment, grid texture, head/hands/feet removed (§3.2) | `SMPL/upperbody_smpl/UpperBody.py`, `SMPL/fullbody_smpl/FullBody.py`, `OffscreenRenderer/flat_renderer.py`, `assets/smpl/*.npz`, `assets/color_pattern/board_300x300.png` | implemented (upper- and full-body templates) |
| Sapiens joint heatmaps $I_{shm}$ (§3.2) | none | missing |
| Garment-invariant representation $I_{gi}$ (Eq. 1) | none | missing |
| BodyMap network (§3.3) | none | missing |
| Simplified body semantic map (§3.3, Fig. 7) | `util/densepose_util.py:IUV2SSDP` (part colours, no UV), used by the full-body path; `IUV2SDP` keeps UV (the [57] map), used by the upper-body path | implemented, from DensePose rather than BodyMap |
| Merged upper/lower torso (§3.3, Fig. 8) | `IUV2SDP` / `IUV2SSDP` paint all torso (and leg) labels white | implemented |
| SAM2 garment masks (§3.4) | external (`training_instructions.md` recommends muggled_sam); masks are read from the alpha channel of exported PNGs | external |
| Per-garment dataset generation (§3.4, Fig. 9) | `DatasetGeneration/upperbody_dataset_generation.py`: **upper body, DensePose maps ([57])**, no sequences | differs (upper body, [57] style) |
| Full-body / loose-garment dataset generation | none | missing |
| Hybrid representation $I_{hybrid}$ (Eq. 2) | `VITON/viton_fullbody_seq.py` (VM + SSDP); `VITON/viton_upperbody.py` (VM + SDP) | implemented |
| ReGarSyn generator: pix2pixHD + ConvLSTM (§3.5) | `model/pix2pixHD/networks.py:RNNGenerator`, `model/ConvLSTM/ConvLSTM.py`, `pix2pixHD_rnn_rgba.py`. The ConvLSTM (2 layers, 3×3, 128 channels) sits after three of the four decoder up-sampling steps, at half the input resolution | implemented |
| Recurrence with $C$/$H$ states (Eq. 3) | `RNNGenerator.inference_forward`, `ConvLSTM.inference_forward`, `reset` | implemented |
| Training on clips (§4.1) | `Pix2PixHD_RNN_RGBA.train_forward`: earlier frames only warm up the state; the loss uses the last 8 | implemented (model side) |
| Recurrent training script and sequence dataset | none | missing |
| Per-frame pix2pixHD loss (Eq. 4) | `pix2pixHD_rgba.py` and `pix2pixHD_rnn_rgba.py`: LSGAN with 2 discriminators, feature matching ($\lambda=10$), VGG ($\lambda=10$, from `--lambda_feat`). [57] describes both weights as 1 | implemented (weights differ; VGG offset bug, see 1.6) |
| "No explicit temporal loss" (§3.5) | `pix2pixHD_rnn_rgba.py:temporal_loss`: RAFT optical-flow L1 loss (weight 1) added; also an L1 loss on the mask ($\lambda=10$). Both are added into the value logged as `G_VGG` | differs |
| Inference with DensePose (§3.6) | `VITON/viton_fullbody_seq.py`, `VITON/viton_upperbody.py` | implemented |
| Composition (Eq. 5) | `composition/naive_overlay.py:naive_overlay_alpha`: alpha thresholded at 128, eroded by a 3×3 kernel, then Eq. 5 with a 0/1 mask | implemented (hard mask) |
| KID / FID / VFID (§4.3) | none | missing |
| Frame-rate measurement (§4.4) | none (no timing code) | missing |
| Frame-difference maps (Fig. 16) | `util/temporal_variance_heatmap.py` (per-pixel variance over a window), not imported anywhere | similar, unused |
| Ablation variants DP / HM / SHM / VM (§4.5) | none | missing |
