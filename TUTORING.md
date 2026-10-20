# RTV tutoring notes

Working notes for learning this repository in depth before changing it. Phase 1 is an audit of the paper against
the code; Phase 2 is the lesson plan. No code is changed during tutoring.

- Paper: Wu, Shen, Igarashi, *Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting
  Garments*, arXiv 2506.12348v3 (2025). Section, figure and equation numbers below refer to it.
- "[57]" is the paper's reference to the earlier per-garment method by the same group (arXiv 2506.10468), which
  most of this repository actually implements (see 1.4).
- Code references are `path:function` on branch `claude/keen-ritchie-fff49z`.

## Progress

| Lesson | Title | Status |
|---|---|---|
| 1 | The big picture: what the system does and how to run it | not started |
| 2 | Getting frames in: camera, orientation, crop, threads | not started |
| 3 | The 3D body: BEV, SMPL, the crop window and the VM render | not started |
| 4 | The 2D body map: DensePose, IUV, SDP and SSDP | not started |
| 5 | Why body maps break under loose clothing: the garment-invariant representation and BodyMap | not started |
| 6 | Making a per-garment dataset | not started |
| 7 | The networks: pix2pixHD, the RGBA output and the ConvLSTM | not started |
| 8 | The losses | not started |
| 9 | The training loop and data loader | not started |
| 10 | Inference and compositing: from network output to the mirror | not started |
| 11 | Measuring it: latency, temporal stability and the paper's metrics | not started |
| 12 | From the audit to our goals | not started |

---

## Phase 1: Paper vs. repository audit

### 1.1 What the paper proposes

**Problem.** Per-garment try-on trains one network per garment on a video of a real person wearing it. The
network learns to draw that garment given a description of the body. The prior method [57] describes the body with
a DensePose body-part map. Two failures motivate the paper:

1. **Loose garments hide the body.** DensePose is accurate on tight clothing but degrades under a hanfu, hanbok or
   dress (Fig. 4). Training data is then generated from degraded maps, while at inference the customer wears tight
   clothing and DensePose is accurate. This train/test distribution gap degrades results (Fig. 2).
2. **Jitter.** [57] synthesises each frame independently, so loose garments flicker even when the person stands
   still.

**Method overview (Fig. 3).** Two networks per setup:

- **BodyMap network** (one per *person*, i.e. per human model who records garments). It maps a
  *garment-invariant representation* to a *body semantic map*. It is trained on a video of the same person in a
  **tight** garment, where DensePose is reliable and serves as ground truth (§3.3).
- **ReGarSyn network** (one per *garment*). A pix2pixHD generator with a ConvLSTM, mapping a hybrid body
  representation to the garment image and mask, frame by frame with a recurrent state (§3.5).

**Garment-invariant representation `I_gi` (§3.2, Fig. 6, Eq. 1).** `I_gi = I_vm ⊕ I_shm`, 6 channels:

- `I_vm`, the *virtual measurement garment* ([58]): the SMPL body fitted by **BEV** with the shape parameters
  β fixed to zero (shape estimates are inflated by loose clothing), head, hands and feet removed, textured with a
  grid pattern ([21]) so orientation is visible, rendered to RGB. A full-body template is used.
- `I_shm`: **Sapiens** 133-keypoint heatmaps, with the hip and knee joints removed (their positions and
  confidences move with loose clothing, Fig. 5), packed into 3 channels: R = right-limb joints, G = head joints,
  B = left-limb joints.

**Body semantic map (§3.3, Fig. 7, 8).** A further simplified DensePose map: continuous UV coordinates removed, and
upper and lower torso merged (their boundary depends on the garment even when tight).

**Per-garment dataset for a loose garment (§3.4, Fig. 9).** For each frame of the video of the human model wearing
the garment: garment image `I_g` and mask `I_m` from **SAM2**; `I_vm`; `I_gi`; then the trained BodyMap network gives
`I_sdp`. Frames keep their order so sequences can be used for training. A separate video of the same model in a
tight garment, doing the same movements, trains the BodyMap network.

**ReGarSyn (§3.5, Fig. 10, Eq. 2–4).**

- Input: hybrid representation `I_hybrid = I_vm ⊕ I_sdp` (6 channels). `I_vm` carries 3D pose (needed for 360°
  views); `I_sdp` gives pixel-level alignment.
- Recurrence (Eq. 3): `Ĩ_g^t, Ĩ_m^t, C^t, H^t = ReGarSyn(I_hybrid^t, C^{t-1}, H^{t-1})`, with C/H the ConvLSTM cell and
  hidden states.
- Loss (Eq. 4): the sum over the clip of pix2pixHD's per-frame loss on `(Ĩ_g ⊕ Ĩ_m)` vs `(I_g ⊕ I_m)`. **No explicit
  temporal loss**: temporally consistent ground truth is said to be enough.
- pix2pixHD's per-frame loss = least-squares GAN loss with multi-scale discriminators + discriminator
  feature-matching loss + VGG perceptual loss.

**Inference (§3.6, Fig. 11, Eq. 5).** The user is assumed to wear tight clothing, so DensePose gives `I_sdp`
directly; BEV gives `I_vm`. ReGarSyn runs with the previous step's states. The result is composited:
`I_o = I ⊙ (1 − Ĩ_m) + Ĩ_g ⊙ Ĩ_m` (Eq. 5).

**Data (§4.1).** Four loose garments (hanfu, down jacket, hanbok, dress), about 3,000 images each, one human model,
plus a tight-garment video of the same model.

**Hyperparameters (§4.1).** BodyMap at 512×384; ReGarSyn at 576×432; 40 epochs per garment; Adam, lr 2e-4,
β = (0.5, 0.999); random training clips of 8–60 frames; about 12 h per garment on an RTX 4090.

**Evaluation (§4.2–4.6).**

- Qualitative comparison with OOTDiffusion, ViViD and [57] (Fig. 12); unseen body shapes (Fig. 13).
- KID, FID and VFID (I3D backbone), computed between the try-on video and a video of the same person really wearing
  the garment, with similar but unsynchronised motion (Table 1). No paired ground truth exists.
- Frame rate on an RTX 4090 (Table 2): [57] 12.38 fps; ours without ConvLSTM 12.15; ours 10.50.
- Ablations: DP / HM / SHM / VM vs the proposed representation, qualitative only (Fig. 14, 15); ConvLSTM on/off
  with frame-difference maps (Fig. 16).

**Stated limitations (§5).** The worn garment is not removed; composition masks are inaccurate; one dataset and
network per garment; tight clothing assumed at inference; no size or fit information.

### 1.2 What is in the repository

| Folder / file | What it is | Used by |
|---|---|---|
| `rtl_demo.py` | Live upper-body try-on app (PyQt5 window, camera thread, garment list) | entry point |
| `demo.py` | Live full-body try-on with the recurrent network (`coat_seq_vmssdp2ta_576`) | entry point |
| `Inference/upperbody_inference.py` | Offline: video file in, `output.mp4` out | entry point |
| `DatasetGeneration/upperbody_dataset_generation.py` | Builds a per-garment upper-body dataset from a video + SAM masks | entry point (training) |
| `Training/upperbody_training.py` | Trains one upper-body garment network | entry point (training) |
| `VITON/viton_upperbody.py` | `FrameProcessor`: per-frame upper-body pipeline | live + offline |
| `VITON/viton_fullbody_seq.py` | `FullBodySeqFrameProcessor`: full-body recurrent pipeline | `demo.py` |
| `VITON/viton_fullbody.py` | Older full-body variant; cannot be imported (missing `ckpt_dict`) | dead |
| `SMPL/` | BEV wrapper (`my_bev.py`), `SMPL_Regressor` (pose, shape, crop transform), numpy SMPL (`smpl_np*.py`, `model.pkl`), upper- and full-body VM templates, smoothing helpers | all pipelines |
| `OffscreenRenderer/` | OpenGL off-screen renderer for the VM image (EGL on Linux, GLFW on Windows) | all pipelines |
| `model/DensePose/` | Vendored DensePose (detectron2 project) and `DensePoseExtractor` | all pipelines |
| `model/pix2pixHD/` | pix2pixHD networks and model wrappers; `pix2pixHD_rgba.py` (upper body), `pix2pixHD_rnn_rgba.py` (recurrent), about 10 unused variants (ASAPNet, dual, mask, align, inpaint, utransformer, ui) | all pipelines |
| `model/ConvLSTM/` | ConvLSTM cell and stack | recurrent model |
| `model/garment_model.py` | Shared checkpoint loader + `rtv_meta.json` (added in this branch) | inference |
| `composition/` | Compositing: `compositor.py` (CPU), `gpu_ops.py` (GPU), `naive_overlay.py` (original) | inference |
| `Datasets/upperbody_garment/` | PyTorch dataset with random-affine augmentation | training |
| `OpticalFlow/` | RAFT (torchvision) optical-flow wrapper | recurrent training loss |
| `options/` | pix2pixHD-style command-line options (base / train / test) | training + model creation |
| `util/` | Image warps, video I/O, DensePose helpers, matting (`matte_util.py`), latency (`latency.py`), GPU check, debug view, many unused helpers | everywhere |
| `assets/` | SMPL templates and UV maps (`assets/smpl`), the grid texture, garment thumbnails, pose guides | everywhere |

**External models and weights the code expects:**

- **Garment checkpoints**: `rtv_ckpts` from Hugging Face (`wuzaiqiang/rtv_ckpts`). Upper body:
  `lab_03/04/07_vmsdp2ta`, `jin_17/18/22_vmsdp2ta` (pix2pixHD, 6→4 channels, 512×512). Full body:
  `coat_seq_vmssdp2ta_576` (recurrent, 576×432). Not in the repository.
- **BEV / ROMP** (`simple_romp` pip package): monocular multi-person SMPL fitting; downloads its own weights.
- **DensePose R50-FPN s1x** (detectron2): weights downloaded from `dl.fbaipublicfiles.com` on first run.
- **SMPL body model**: `SMPL/model.pkl` and the meshes in `assets/smpl`.
- **RAFT-small** (torchvision weights): only for the recurrent training loss.
- **VGG19** (torchvision weights): pix2pixHD's perceptual loss, training only.
- Libraries: PyTorch, detectron2, OpenCV, PyQt5, PyOpenGL (+ glfw on Windows), trimesh, PyGLM, scipy.

### 1.3 Paper component → code

| Paper component (section) | Where in the repository | Status |
|---|---|---|
| BEV SMPL fit (§3.2) | `SMPL/smpl_regressor.py:SMPL_Regressor`, `SMPL/my_bev.py:MyBEV` | implemented |
| β fixed to zero (§3.2) | `MyBEV.forward` with `fix_body`; zeroes betas 0–3 and 10 only (`'all'` zeroes all) | differs (partial) |
| VM garment, grid texture, head/hands/feet removed (§3.2) | `SMPL/upperbody_smpl/UpperBody.py`, `SMPL/fullbody_smpl/FullBody.py`, `OffscreenRenderer/flat_renderer.py`, `assets/smpl/*.npz`, `assets/color_pattern/board_300x300.png` | implemented (upper- and full-body templates) |
| Sapiens joint heatmaps `I_shm` (§3.2) | none | missing |
| Garment-invariant representation `I_gi` (Eq. 1) | none | missing |
| BodyMap network (§3.3) | none | missing |
| Simplified body semantic map (§3.3, Fig. 7) | `util/densepose_util.py:IUV2SSDP` (part colours, no UV), used by the full-body path; `IUV2SDP` keeps UV (the [57] map), used by the upper-body path | implemented, from DensePose rather than BodyMap |
| Merged upper/lower torso (§3.3, Fig. 8) | `IUV2SDP` / `IUV2SSDP` paint all torso (and leg) labels the same colour | implemented |
| SAM2 garment masks (§3.4) | external (`training_instructions.md` recommends muggled_sam); masks are read from PNGs | external |
| Per-garment dataset generation (§3.4, Fig. 9) | `DatasetGeneration/upperbody_dataset_generation.py`: **upper body, DensePose maps ([57])**, no sequences | differs (upper body, [57] style) |
| Full-body / loose-garment dataset generation | none | missing |
| Hybrid representation `I_hybrid` (Eq. 2) | `VITON/viton_fullbody_seq.py` (VM + SSDP); `VITON/viton_upperbody.py` (VM + SDP) | implemented |
| ReGarSyn generator: pix2pixHD + ConvLSTM (§3.5) | `model/pix2pixHD/networks.py:RNNGenerator`, `model/ConvLSTM/ConvLSTM.py`, `pix2pixHD_rnn_rgba.py` | implemented |
| Recurrence with C/H states (Eq. 3) | `RNNGenerator.inference_forward`, `ConvLSTM.inference_forward`, `reset` | implemented |
| Training on clips (§4.1) | `Pix2PixHD_RNN_RGBA.train_forward`: earlier frames only warm up the state; the loss uses the last 8 | implemented (model side) |
| Recurrent training script and sequence dataset | none | missing |
| Per-frame pix2pixHD loss (Eq. 4) | `pix2pixHD_rgba.py` and `pix2pixHD_rnn_rgba.py`: LSGAN with 2 discriminators, feature matching (λ = 10), VGG (λ = 10) | implemented |
| "No explicit temporal loss" (§3.5) | `pix2pixHD_rnn_rgba.py:temporal_loss`: RAFT optical-flow L1 loss added; also an L1 loss on the mask | differs |
| Inference with DensePose (§3.6) | `VITON/viton_fullbody_seq.py`, `VITON/viton_upperbody.py` | implemented |
| Composition (Eq. 5) | `composition/compositor.py`, `composition/gpu_ops.py:compose` (soft); `naive_overlay.py` (original hard mask) | implemented (extended) |
| KID / FID / VFID (§4.3) | none | missing |
| Frame-rate measurement (§4.4) | `util/latency.py` (added in this branch) | added |
| Frame-difference maps (Fig. 16) | `util/temporal_variance_heatmap.py` (per-pixel variance over a window) | similar |
| Ablation variants DP / HM / SHM / VM (§4.5) | none | missing |

### 1.4 Implemented as in the paper, differently, missing, extra

**As in the paper**

- BEV-based SMPL fitting and the grid-textured VM render, including the full-body template the paper uses.
- The simplified body map without UV and with merged torso (`IUV2SSDP`), used by the full-body pipeline.
- The ReGarSyn architecture (pix2pixHD with a ConvLSTM in the decoder) and its recurrent inference.
- pix2pixHD's per-frame losses and the optimiser settings (Adam, lr 2e-4, β1 = 0.5).
- Inference from DensePose assuming tight clothing, and composition with the predicted mask.

**Implemented differently**

- **Two methods in one repository.** The upper-body path (`rtl_demo.py`, dataset generation, training) is the
  earlier method [57]: upper-body VM + DensePose SDP *with* UV, one image at a time, 512×512 crops. Only the
  full-body inference path (`demo.py`) is the paper's method.
- **β fixed to zero** zeroes only betas 0–3 and 10, not all; and the live upper-body demo uses the customer's
  estimated shape by default (`--body_shape`).
- **Training loss**: the recurrent model adds an optical-flow (RAFT) temporal loss and an L1 mask loss, against the
  paper's statement of no temporal loss. The VGG loss is on the unmasked colour for the recurrent model but on
  masked colour for the upper-body model.
- **Gradient through time**: only the last 8 frames of a clip get gradients; earlier frames warm the state without
  gradients (truncated back-propagation through time). The paper only says clips of 8–60 frames.
- **Body semantic map at training time** comes from DensePose (upper-body generation), not from BodyMap.
- **Crop window**: upper body uses a square 512 crop centred near the upper chest and sized from the
  pelvis-to-head distance; full body a 576×432 crop centred on the pelvis and sized from the head-to-feet distance
  (`SMPL_Regressor.get_trans2roi` / `get_fullbody_trans2roi`).

**Missing**

- Sapiens joint heatmaps, the garment-invariant representation, and the BodyMap network (the paper's first
  contribution).
- Full-body / loose-garment dataset generation (§3.4) and a training script and sequence dataset for ReGarSyn.
- Evaluation code for KID, FID and VFID, and the ablation variants.
- Garment removal (also listed as a limitation in the paper).

**In the repository but not in the paper**

- The upper-body [57] pipeline, Qt demo app and offline video script.
- About 10 unused pix2pixHD variants and about 19 unused `util/` files.
- Vendored DensePose/detectron2 project code; RAFT and VGG19 weights from torchvision.
- Added in this branch: latency instrumentation, GPU check, debug view, Windows/GLFW renderer, robust camera
  opening (DroidCam), soft compositing (on GPU), hand/head occlusion, person matching between BEV and DensePose,
  missed-frame hold, exposure lock, straight-colour training targets, checkpoint metadata. See the git log.

### 1.5 Where we would change things for our goals

| Goal | Most likely places | Why |
|---|---|---|
| **Less jitter and inconsistency** | Use the recurrent model for upper body too (`pix2pixHD_rnn_rgba.py`, `RNNGenerator`) and write its training script; crop smoothing (`util/cv2_trans_util.py:TemporalSmoothing`); pose filtering (`SMPL/pose_filter.py`, unused); BEV temporal settings (`SMPL_Regressor.create_bev_model`) | The upper-body networks are per-frame, so they flicker by design (§1); the crop window and the body fit are noisy frame to frame. |
| **Higher fps, lower latency** | DensePose (largest stage) → render the body map from the SMPL fit instead; DensePose input size; fp16 / TensorRT for the generator; run BEV and DensePose concurrently (`VITON/viton_upperbody.py`) | Measured: BEV ~32 ms and DensePose ~50 ms dominate; the generator is ~10–16 ms. |
| **Our own garments** | `DatasetGeneration/upperbody_dataset_generation.py`, `Datasets/upperbody_garment`, `Training/upperbody_training.py`, `training_instructions.md`; fine-tuning from an existing checkpoint (`--load_pretrain`) | This pipeline already trains new upper-body garments from a phone video + SAM masks; fine-tuning cuts the ~12 h per garment. |
| **Outfit recommendation** | New module, outside the try-on pipeline (e.g. `recommend/`), plus garment metadata next to each checkpoint (`rtv_meta.json`) and the garment list in `rtl_demo.py` | Nothing in the paper or code covers it; it needs garment attributes or image embeddings and a compatibility model or rules. |
| **Lower-body garments (pants, shorts)** | A lower-body VM template (like `SMPL/upperbody_smpl`), a lower-body crop window (`SMPL_Regressor`), dataset generation and training variants, and a second network composited under/over the upper garment (`VITON/`, `composition/`) | Only upper-body and full-body (dress/coat) templates and pipelines exist; pants need leg parts, a different crop, and layering with the top. |
| **Robust to background noise** | Person selection (`DensePoseExtractor.select_person`, `SMPL_Regressor` depth order); a customer tracker / mirror zone; person segmentation before compositing; exposure lock (`rtl_demo.py`) | Bystanders and clutter corrupt BEV and DensePose; both models now agree on one person, but nothing tracks the customer across frames yet. |

---

## Phase 2: Lesson plan

Lessons follow the data: a camera frame (or a training video) enters, becomes body representations, passes
through the network, and leaves as a composited image or a trained checkpoint, then gets measured. Each lesson
starts with intuition, walks the code step by step (every import and external model explained, tensor shapes at
each step), introduces any maths in plain words first, ties back to the paper, and ends with a recap and 2–3
questions.

### Lesson 1: The big picture: what the system does and how to run it
- **Files:** `README.md`, `training_instructions.md`, `rtl_demo.py` (top level only), `demo.py`,
  `Inference/upperbody_inference.py`, `VITON/viton_upperbody.py` (`__call__` outline only)
- **Key concepts:** per-garment vs general try-on; training-time vs run-time pipelines; the two methods in this repo
  ([57] upper body vs the paper's full body); what a "garment checkpoint" is; the stages of one frame.
- **Paper:** §1, §3.1, Fig. 2, 3, 11.

### Lesson 2: Getting frames in: camera, orientation, crop, threads
- **Files:** `rtl_demo.py` (`VitonThread`: `get_camera`, `grab_frames`, `run`), `util/image_warp.py`
  (`crop2_169`), `util/camera_util.py`, `util/gpu_check.py`
- **Key concepts:** OpenCV `VideoCapture` and backends (DirectShow, FFmpeg streams); BGR images as `(H, W, 3)` uint8
  arrays; rotation and mirroring; 9:16 crop and the 576×1024 working size; producer/consumer threads and why only
  the newest frame is kept; Qt signals.
- **Paper:** §3.6 (inference input), §4.4 (real time).

### Lesson 3: The 3D body: BEV, SMPL, the crop window and the VM render
- **Files:** `SMPL/smpl_regressor.py`, `SMPL/my_bev.py`, `SMPL/smpl_np_base.py`,
  `SMPL/upperbody_smpl/UpperBody.py`, `SMPL/fullbody_smpl/FullBody.py`, `OffscreenRenderer/gl_utils.py`,
  `flat_renderer.py`, `off_screen_render.py`, `shaders.py`, `assets/smpl/*`, `util/cv2_trans_util.py`
- **Key concepts:** the SMPL model (pose θ, shape β, 6,890 vertices) and what BEV estimates; camera translation and
  perspective projection; choosing the person (depth order); the affine crop transform `trans2roi` and its inverse;
  rendering a textured mesh off-screen with OpenGL; why a grid texture shows orientation; smoothing the crop over
  time.
- **Paper:** §3.2 (VM part, β = 0), [58].

### Lesson 4: The 2D body map: DensePose, IUV, SDP and SSDP
- **Files:** `model/DensePose/densepose_extractor.py` (`get_IUV`, `select_person`), `model/DensePose/apply_net.py`
  (config), `model/DensePose/configs/*`, `util/densepose_util.py` (`IUV2SDP`, `IUV2SSDP`, `IUV2HandHeadMask`)
- **Key concepts:** detectron2 and Mask R-CNN-style detection; DensePose's per-pixel part index I (24 parts) and
  surface coordinates U, V; the IUV image `(H, W, 3)`; why the torso is painted white; SDP (with UV) vs SSDP
  (without); detection scores and choosing the same person as BEV.
- **Paper:** §3.3, Fig. 4, 7, 8.

### Lesson 5: Why body maps break under loose clothing: the garment-invariant representation and BodyMap
- **Files:** paper only (not in the repo); compare with `SMPL/my_bev.py` (`fix_body`) and Lesson 4's maps.
- **Key concepts:** distribution shift between training and inference; garment-sensitive joints; joint heatmaps
  (Gaussian blobs per joint); the 6-channel `I_gi`; a per-person image-to-image network as a "translator"; where it
  would plug into our code and what it costs at run time.
- **Paper:** §2.4, §3.2, §3.3, Fig. 2, 5, 6, 14, 15, Eq. 1.

### Lesson 6: Making a per-garment dataset
- **Files:** `DatasetGeneration/upperbody_dataset_generation.py`, `DatasetGeneration/options.py`,
  `util/multithread_video_loader.py`, `util/file_io.py`, `training_instructions.md`
- **Key concepts:** recording protocol; SAM garment masks; what each saved file holds (`_garment.png`, `_vm.jpg`,
  `_mask.png`, `_iuv.npy`, `_trans2roi.npy`, `dataset_info.json`); ground truth = the garment pixels from the same
  frame; what the paper's loose-garment version adds (BodyMap maps, sequences).
- **Paper:** §3.4, Fig. 9, §4.1 (datasets).

### Lesson 7: The networks: pix2pixHD, the RGBA output and the ConvLSTM
- **Files:** `model/pix2pixHD/networks.py` (`GlobalGenerator`, `RNNGenerator`, `MultiscaleDiscriminator`,
  `NLayerDiscriminator`, `ResnetBlock`), `model/pix2pixHD/pix2pixHD_model.py`, `pix2pixHD_rgba.py`, `pix2pixHD_rnn_rgba.py`,
  `base_model.py`, `models.py`, `model/ConvLSTM/ConvLSTM.py`, `model/garment_model.py`
- **Key concepts:** encoder / residual blocks / decoder; tensor shapes through the generator
  (`(1, 6, 512, 512)` → ... → `(1, 4, 512, 512)`); Tanh outputs; why the output is RGB + alpha; instance
  normalisation; what an LSTM gate does and how a ConvLSTM applies it to feature maps; where the ConvLSTM sits;
  hidden and cell states across frames; loading checkpoints.
- **Paper:** §3.5, Eq. 2, 3, Fig. 10, [56] (pix2pixHD).

### Lesson 8: The losses
- **Files:** `pix2pixHD_rgba.py:forward`, `pix2pixHD_rnn_rgba.py:train_forward` / `temporal_loss`,
  `networks.py` (`GANLoss`, `VGGLoss`, `Vgg19`), `OpticalFlow/optical_flow.py`
- **Key concepts:** adversarial training (generator vs discriminator); least-squares GAN loss; multi-scale
  discriminators; feature matching; perceptual (VGG) loss; masked vs unmasked colour loss; the mask L1 loss; optical
  flow and the RAFT temporal loss the code adds; loss weights (λ = 10).
- **Paper:** §3.5, Eq. 4; differences listed in 1.4.

### Lesson 9: The training loop and data loader
- **Files:** `Training/upperbody_training.py`, `Datasets/upperbody_garment/upperbody_garment.py`,
  `options/base_options.py`, `train_options.py`, `test_options.py`, `util/visualizer.py`, `util/matte_util.py`
- **Key concepts:** PyTorch `Dataset` / `DataLoader`; random affine augmentation of the crop; straight vs
  premultiplied colour; alternating generator and discriminator updates; epochs, `niter` / `niter_decay` and
  linear learning-rate decay; checkpoints and `rtv_meta.json`; fine-tuning with `--load_pretrain`; what a
  recurrent training script would add (clips, warm-up, state reset).
- **Paper:** §4.1 (implementation details).

### Lesson 10: Inference and compositing: from network output to the mirror
- **Files:** `VITON/viton_upperbody.py` (full walk-through), `VITON/viton_fullbody_seq.py`,
  `composition/gpu_ops.py`, `composition/compositor.py`, `composition/naive_overlay.py`, `util/debug_view.py`
- **Key concepts:** warping network inputs into the crop and outputs back (affine warps, `grid_sample`); the
  6-channel input built per frame; turning Tanh output into colour and alpha; alpha compositing (Eq. 5);
  premultiplied colour and soft vs hard masks; keeping hands and face in front; holding the garment on missed
  frames; recurrent state at inference; GPU vs CPU work.
- **Paper:** §3.6, Eq. 5, Fig. 11, §5 (composition limitations).

### Lesson 11: Measuring it: latency, temporal stability and the paper's metrics
- **Files:** `util/latency.py`, `util/temporal_variance_heatmap.py`, `util/debug_view.py`; the paper's evaluation
  (no code in the repo)
- **Key concepts:** throughput (fps) vs latency (camera-to-display); timing GPU work correctly; frame-difference and
  temporal-variance maps; FID, KID and VFID in plain terms (comparing feature statistics of real and generated
  images or videos) and why they need care with small test sets; designing our own test set.
- **Paper:** §4.3–4.6, Tables 1, 2, Fig. 16.

### Lesson 12: From the audit to our goals
- **Files:** revisit 1.5; `SMPL/pose_filter.py`, `SMPL/upperbody_smpl` vs `fullbody_smpl` (templates for lower
  body), `rtl_demo.py` garment list
- **Key concepts:** turning each goal (jitter, speed, own garments, recommendation, lower body, background
  robustness) into a concrete change, a measurement, and an order of work.
- **Paper:** §5 (limitations and future work).
