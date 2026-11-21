---
title: Hazards before editing
summary: Conventions and bugs that silently break things: colour order, value ranges, sizes, relative paths, ConvLSTM state, threads, known bugs, dead code.
---

# Hazards to know before editing

Written for upstream RTV. Our fork fixes some of them (Windows renderer, VGG offset, label resize, RAFT at inference); check `fork_changes` and the code before assuming a hazard still applies.

### Conventions that silently break things when mixed up
- **Colour order.** Camera frames are BGR (OpenCV). The VM render is **RGB** (`glReadPixels(..., GL_RGB)`, texture
  loaded with PIL). Dataset generation writes it with `cv2.imwrite`, which treats the array as BGR, and the dataset
  loader reads it back with PIL; the network was therefore trained on the render with its channels reversed. That is
  why inference does `vm_tensor[:, [2, 1, 0]]`. The body map is *never* swapped: its three channels are I, U, V and
  are fed as if they were R, G, B, in training and inference alike. The network outputs RGB colour;
  `tensor2im(..., rgb=False)` turns it into BGR for compositing.
- **Value ranges.** Network inputs and RGB outputs are in $[-1,1]$ (`x / 255 * 2 - 1`). Alpha is not: see the alpha
  bullet in 1.4. Images outside the networks are `uint8` in $[0,255]$.
- **Image sizes.** Camera frames are rotated 90° clockwise, mirrored, resized to 1024 high and cropped to 9:16, so a
  16:9 webcam gives 576 × 1024 (width × height). Upper-body networks see 512 × 512 crops; the full-body network sees
  576 × 432 (height × width).
- **Affine matrices** are 2 × 3 OpenCV matrices that map camera pixels to crop pixels (`trans2roi`); the inverse maps
  back. Anything that changes the crop changes both.

### State and process hazards
- **Relative paths.** `./assets/...`, `./model/DensePose/...`, `./rtv_ckpts`, `./checkpoints` and
  `./PerGarmentDatasets` are resolved from the current directory, so every script must be run from the repository
  root.
- **Linux only.** The renderer forces EGL (`PYOPENGL_PLATFORM=egl`), which does not exist on Windows.
- **The renderer's size is fixed at its first call.** `UpperBodySMPL.render` creates the `FlatRenderer` lazily with
  the first frame's size; later frames of a different size are rendered at the old size.
- **ConvLSTM state lives inside the module** (`self.h`, `self.c`) and is only cleared by `reset()`, which
  `FullBodySeqFrameProcessor` never calls: the state carries over between people and after BEV misses. One model
  instance can serve one video stream. `ConvLSTM.inference_forward` returns `x[-1].unsqueeze(0)`, the last item of
  the batch, so inference only works with batch size 1.
- **Garment switching runs on background threads** (`set_target_garment` → `switch_to_target_garment`) and is
  guarded by one lock. All checkpoints are loaded to CPU at start-up by another thread and moved to the GPU when
  selected. If the generator call raised an exception, the lock would never be released and the app would freeze.
- **Training and inference use different shape settings** (`fix_body`), see 1.4.
- **Bugs that look harmless:**
  - `pix2pixHD_rgba.py` and `pix2pixHD_rnn_rgba.py` compute the VGG loss on `x * 0.5 + 1.0`, mapping $[-1,1]$ to
    $[0.5,1.5]$ instead of $[0,1]$ (`+0.5` was intended). The released checkpoints were presumably trained this way.
  - `Datasets/upperbody_garment/upperbody_garment.py` calls `cv2.resize(dp_img, (raw_w, raw_h), cv2.INTER_NEAREST)`.
    The third positional argument of `cv2.resize` is `dst`, not the interpolation, so part labels are resized
    bilinearly and their boundaries get blended values. Pass `interpolation=cv2.INTER_NEAREST`.
  - In `rtl_demo.py`, row 0 of the list is "None" and rows 1 to 6 are the garments, but `keyPressEvent` handles only
    keys 0 to 5, so the sixth garment needs a click (or key 6 while the list itself has focus, via `eventFilter`). The
    `gid_map` dictionary is never used.
- **Stale or dead code that looks live:** `VITON/viton_fullbody.py` (import fails), the "36x27" comment in
  `RNNGenerator`, the `weights_init` guard that tries to skip LSTM layers by class name (the ConvLSTM's
  convolutions are plain `Conv2d`, so they are initialised like the rest), `SMPL_Regressor.__init__`'s
  `beta[1] += 1.2` ("todo:???"), which the BEV path never uses, and `util/camera_util.list_available_cameras`, imported
  by `rtl_demo.py` but never called.

**How the ConvLSTM placement was verified.** `networks.py` gives the generator `n_downsample_global = 4` (default in
`options/base_options.py`). Counting parameters by hand for the plain generator (6 input channels, 4 output channels,
$n_{gf}=64$, 9 residual blocks) gives 182,455,812 weights, i.e. 729,823,248 bytes in fp32. The released
`lab_03_vmsdp2ta/latest_net_G.pth` is 729,845,274 bytes (the extra 22 KB is file overhead). The recurrent checkpoints
are 9,443,144 bytes larger. A ConvLSTM with 2 layers, 3 × 3 kernels and 128 channels has
$2\,(256\cdot512\cdot9+512)=2{,}360{,}320$ weights, i.e. 9,441,280 bytes. A ConvLSTM at the 1024-channel bottleneck would
add about 604 MB, so this is the only placement consistent with the files. This matches the code: in
`RNNGenerator.__init__` the ConvLSTM is created in the last up-sampling iteration, where
$n_{gf}\cdot 2 = 128$.
