---
title: Generator, losses and training
summary: pix2pixHD generator facts (182M weights, 93 percent in residual blocks), the ConvLSTM, the losses actually used, training commands and settings, and what is missing for recurrent training.
---

# Generator, losses and training

## Generator (`model/pix2pixHD/networks.py`)

- pix2pixHD `GlobalGenerator`: 6 input channels (VM + body map), 4 output channels (RGB + alpha), $n_{gf} = 64$,
  4 down-sampling steps, 9 residual blocks at 1024 channels.
- About **182M weights (730 MB fp32)**; the **9 residual blocks hold about 93 percent** of weights and most compute,
  so they are the place to slim for FPS and the natural shared backbone (M5).
- `RNNGenerator` (recurrent): ConvLSTM with 2 layers, 3 x 3 kernels, 128 channels, after three of the four decoder
  up-sampling steps (half resolution). Adds about 2.4M weights. State lives in the module (`self.h`, `self.c`);
  `reset()` clears it; inference works only with batch size 1.
- Checkpoint names: `vm` + `sdp`/`ssdp` inputs, `2ta` = to texture and alpha, `seq` = recurrent, `576` = crop height.

## Losses actually used

- LSGAN with 2 multi-scale PatchGAN discriminators (about 5.5M weights), feature matching ($\lambda = 10$), VGG19
  perceptual ($\lambda = 10$, from `--lambda_feat`). The earlier paper [57] describes weights of 1.
- Upper-body model (`pix2pixHD_rgba.py`): VGG on masked colour, **no mask loss** (alpha learned only through GAN,
  feature matching and masked VGG).
- Recurrent model (`pix2pixHD_rnn_rgba.py`): adds an L1 mask loss ($\lambda = 10$) and a RAFT optical-flow temporal
  loss (weight 1), although the paper says no temporal loss. Both are logged inside `G_VGG`.
- Bug upstream: VGG input `x * 0.5 + 1.0` maps to $[0.5, 1.5]$ instead of $[0, 1]$ (fixed to `+0.5` in our fork).

## Training settings and commands (upper body, [57] method)

```
python DatasetGeneration/upperbody_dataset_generation.py --video_path <video> --mask_dir <SAM mask dir> --dataset_name <garment>
python Training/upperbody_training.py --model pix2pixHD_RGBA --input_nc 6 --output_nc=4 --batchSize 4 --img_size 512 --dataset_path ./PerGarmentDatasets/<garment> --name <garment> --niter 80 --niter_decay 80
python Inference/upperbody_inference.py --input_video <video> --garment_name <garment>
```

- 80 + 80 (decay) epochs, batch 4, Adam optimiser with learning rate $2 \times 10^{-4}$ and $\beta = (0.5, 0.999)$ (pix2pixHD defaults; the paper gives no reason). The 2025 paper: 40 epochs, about
  12 h per garment on an RTX 4090, clips of 8 to 60 frames. [57]: 100 epochs, batch 8, about 15 h on an RTX A6000.
- About 3,000 frames per garment from a 4K phone video; SAM2 garment masks (muggled_sam recommended).
- Useful flags: `--load_pretrain` (fine-tune from a checkpoint), `--dataset_path a,b` (join datasets),
  `--continue_train` (restarts the epoch count at 1).
- Scripts must run from the repo root (relative paths).

## Recurrent training: close to available, not released

- Present: `Pix2PixHD_RNN_RGBA.train_forward` warms the state on early frames and puts the loss on the last 8
  (truncated back-propagation through time); `reset()`; `util/random_affine_batch.py:RandomAffineMatrix.batch_forward`
  (unused) applies one shared augmentation to a whole clip.
- Missing: a clip (sequence) dataset class, a training script, gap handling for skipped frames, and the full-body /
  loose-garment dataset generation. BodyMap code and weights and Sapiens heatmap packing are also not released.
- A recurrent upper-body network on the existing upper-body datasets is a quick first milestone for jitter.

## Memory (estimates, not measured)

- pix2pixHD's README asks for an 11 GB+ GPU at 1024 x 512 batch 1. RTV's batch 4 at 512 x 512 roughly doubles the pixels
  per step. Recurrent clip training and a multi-garment backbone are estimated at 40 to 80 GB. Measure with a short
  pilot run before planning.
