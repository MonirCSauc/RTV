---
title: Glossary
summary: Short definitions of the project's terms and abbreviations (VM, SDP, SSDP, IUV, BEV, SMPL, BodyMap, ReGarSyn, FiLM, RGBA, and more).
---

# Glossary

- **RTV**: Wu, Shen, Igarashi (2025) real-time per-garment try-on repository; our starting code.
- **[57]**: the earlier per-garment method by the same group (arXiv 2506.10468); the upper-body path in the repo.
- **Per-garment network**: one generator trained for one garment from a video of a person wearing it.
- **SMPL**: parametric 3D human body model (pose $\theta$, shape $\beta$).
- **BEV**: monocular multi-person SMPL estimator (from the ROMP family) that fits SMPL to each frame.
- **VM / $I_{vm}$**: virtual measurement garment; the SMPL body without head, hands and feet, textured with a noise
  pattern and rendered to an image. Carries 3D pose and orientation.
- **DensePose / IUV**: per-pixel body-part index (I) and surface coordinates (U, V) predicted from the image.
- **SDP**: body map from DensePose with UV kept (upper-body path).
- **SSDP**: simplified DensePose map: part colours only, no UV, torso merged (full-body path, paper's body map).
- **Sapiens**: a large human-centric vision model; the paper uses its 133 keypoint heatmaps.
- **Garment-invariant representation $I_{gi}$**: VM + Sapiens heatmaps without hips and knees.
- **BodyMap**: per-person network mapping $I_{gi}$ to a clean body map; used only to build training data.
- **ReGarSyn**: the paper's per-garment generator (pix2pixHD + ConvLSTM) producing garment + mask.
- **Hybrid representation**: VM stacked with the body map (6 channels), the generator input.
- **pix2pixHD**: conditional GAN for image-to-image translation (global generator + multi-scale discriminators).
- **LSGAN**: least-squares GAN loss. **Feature matching**: L1 between discriminator features of real and fake.
- **VGG loss**: perceptual loss on VGG19 features.
- **ConvLSTM**: LSTM with convolutions, keeps a spatial memory between frames (reduces flicker).
- **RGBA output**: generator outputs garment colour (RGB) and an alpha mask (A) used for compositing.
- **Soft vs hard alpha**: soft keeps values between 0 and 1 at edges (smooth blend); hard thresholds to 0 or 1.
- **Compositing**: pasting the generated garment onto the camera frame using alpha.
- **SAM2**: segmentation model used to cut the garment out of each recorded frame.
- **RAFT**: optical-flow network; used for the recurrent model's temporal loss.
- **FiLM**: feature-wise linear modulation, $h' = \gamma \odot h + \beta$; a garment code scales and shifts features.
- **Shared backbone**: encoder + residual blocks trained on all garments; per-garment parts specialise.
- **BEDLAM**: synthetic dataset of clothed animated bodies with ground-truth SMPL.
- **Mirror zone**: region in front of the mirror where a person counts as the customer.
- **Held-out chunk**: last part of each recording, never trained on, used for testing.
- **M1 to M5**: the proposal's recommended modifications (topic `proposals`).
