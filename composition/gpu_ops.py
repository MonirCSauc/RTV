"""GPU versions of the per-frame pixel work, so full-size frames do not go through numpy.

Each function mirrors a CPU function and is checked against it (same results up to rounding):
warp_affine ~ cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT 0), iuv_to_sdp ~ util.densepose_util.IUV2SDP,
hand_head_mask ~ util.densepose_util.IUV2HandHeadMask, compose ~ composition.compositor.GarmentCompositor.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_grids = {}


def warp_affine(src, M, out_h, out_w, mode='bilinear'):
    """Like cv2.warpAffine(src, M, (out_w, out_h)): src (N, C, H, W) float, M (2x3) maps src to output pixels.
    mode: 'bilinear' (INTER_LINEAR) or 'nearest' (INTER_NEAREST)."""
    key = (out_h, out_w, src.device)
    if key not in _grids:
        ys, xs = torch.meshgrid(torch.arange(out_h, device=src.device, dtype=torch.float32),
                                torch.arange(out_w, device=src.device, dtype=torch.float32), indexing='ij')
        _grids[key] = (xs, ys)
    xs, ys = _grids[key]
    inv = cv2.invertAffineTransform(np.asarray(M, np.float64))  # output pixel -> source pixel
    sx = float(inv[0, 0]) * xs + float(inv[0, 1]) * ys + float(inv[0, 2])
    sy = float(inv[1, 0]) * xs + float(inv[1, 1]) * ys + float(inv[1, 2])
    h, w = src.shape[-2:]
    # align_corners=True: -1 and 1 are the centres of the first and last source pixels, as in OpenCV
    grid = torch.stack([2.0 * sx / (w - 1) - 1.0, 2.0 * sy / (h - 1) - 1.0], -1)[None].expand(src.shape[0], -1, -1, -1)
    return F.grid_sample(src, grid, mode=mode, padding_mode='zeros', align_corners=True)


_TORSO_LEG = None


def iuv_to_sdp(iuv):
    """IUV2SDP on the GPU. iuv: uint8 (H, W, 3) tensor -> float (1, 3, H, W) with values 0..255."""
    global _TORSO_LEG
    if _TORSO_LEG is None or _TORSO_LEG.device != iuv.device:
        _TORSO_LEG = torch.zeros(256, dtype=torch.bool, device=iuv.device)
        _TORSO_LEG[[1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14]] = True
    sdp = iuv.float()
    sdp[..., 0] = torch.floor(sdp[..., 0] / 24 * 255)  # numpy: float -> uint8 truncates
    sdp[_TORSO_LEG[iuv[..., 0].long()]] = 255
    return sdp.permute(2, 0, 1)[None]


_SSDP_LUT = None


def iuv_to_ssdp(iuv):
    """IUV2SSDP on the GPU: the body map without UV, one flat colour per body part group (paper Fig. 7d).
    iuv: uint8 (H, W, 3) tensor -> float (1, 3, H, W) with values 0..255."""
    global _SSDP_LUT
    if _SSDP_LUT is None or _SSDP_LUT.device != iuv.device:
        lut = torch.zeros(256, 3, device=iuv.device)
        lut[[1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 23, 24]] = 255.0          # torso, legs, feet, head
        lut[[15, 17]] = torch.tensor([255.0, 0.0, 0.0], device=iuv.device)     # left upper arm
        lut[[19, 21]] = torch.tensor([255.0, 255.0, 0.0], device=iuv.device)   # left lower arm
        lut[[4]] = torch.tensor([255.0, 0.0, 255.0], device=iuv.device)        # left hand
        lut[[16, 18]] = torch.tensor([0.0, 0.0, 255.0], device=iuv.device)     # right upper arm
        lut[[20, 22]] = torch.tensor([0.0, 255.0, 255.0], device=iuv.device)   # right lower arm
        lut[[3]] = torch.tensor([0.0, 255.0, 0.0], device=iuv.device)          # right hand
        _SSDP_LUT = lut
    return _SSDP_LUT[iuv[..., 0].long()].permute(2, 0, 1)[None]


def temporal_filter(current, previous, threshold=0.12, floor=0.25):
    """Suppress frame-to-frame flicker of the garment network's output without smearing real motion.

    current, previous: (1, C, H, W) network outputs in the body-centred crop. Where the two differ little (the
    small, random changes a per-frame network makes for a nearly unchanged input), the output moves only part of
    the way to the new value; where they differ a lot (the body moved), the new value is taken as is.
    threshold: difference (in network output units, range 2) treated as real change; floor: the smallest step.
    """
    if previous is None or previous.shape != current.shape:
        return current
    diff = (current - previous).abs().amax(1, keepdim=True)
    diff = F.avg_pool2d(diff, 5, stride=1, padding=2)  # judge change over a small neighbourhood, not single pixels
    k = (diff / threshold).clamp(floor, 1.0)
    return previous + k * (current - previous)


_ELLIPSE = {}


# DensePose part labels kept in front of the garment (see --keep_in_front)
KEEP_IN_FRONT = {'hands_head': (3, 4, 23, 24), 'hands': (3, 4), 'none': ()}


def hand_head_mask(iuv, dilate_px=2, parts=(3, 4, 23, 24)):
    """IUV2HandHeadMask on the GPU: float (H, W) in {0, 1}, dilated with OpenCV's elliptical kernel."""
    label = iuv[..., 0]
    mask = torch.zeros_like(label, dtype=torch.bool)
    for part in parts:
        mask |= label == part
    mask = mask.float()[None, None]
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        if (k, iuv.device) not in _ELLIPSE:
            kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32))
            _ELLIPSE[(k, iuv.device)] = kernel.to(iuv.device)[None, None]
        mask = (F.conv2d(mask, _ELLIPSE[(k, iuv.device)], padding=dilate_px) > 0).float()
    return mask[0, 0]


def garment_support(roi_vm, roi_sdp, radius=30, feather=7):
    """Where a garment may appear in the network's 512 crop: near the VM render (the upper-body template, a
    long-sleeved top) and near the body DensePose sees. Returns float (1, 1, H, W) in 0..1.

    roi_vm, roi_sdp: (1, 3, H, W) network inputs (0..255). A network outside the poses it was trained on (e.g.
    lab_03 with the arm raised on one side) can output alpha all over the crop. An upper-body garment lies on the
    torso and arms, which the VM render marks, so alpha over the legs, the head or the background is cut. The VM
    render comes from the 3D body fit, which can sit a little off the real body, so its margin is wider than the
    DensePose one; both must hold.
    """
    def grown(mask, r):
        k = 2 * r + 1  # square dilation, separable
        mask = F.max_pool2d(mask, (1, k), stride=1, padding=(0, r))
        return F.max_pool2d(mask, (k, 1), stride=1, padding=(r, 0))
    vm = (roi_vm.amax(1, keepdim=True) > 0).float()
    body = (vm.bool() | (roi_sdp.amax(1, keepdim=True) > 0)).float()
    support = grown(body, radius) * grown(vm, radius + radius // 2)
    if feather > 0:  # soft edge, so a garment touching the limit fades instead of being cut
        f = 2 * feather + 1
        support = F.avg_pool2d(support, f, stride=1, padding=feather, count_include_pad=False)
    return support


def garment_core(alpha, roi_vm, threshold=0.85, grow=8, min_share=0.03):
    """Keep only the garment the body wears: confident regions (alpha > threshold) that touch the VM body render,
    grown by `grow` px for the soft edge. alpha: (1, 1, H, W); returns float (1, 1, H, W) in {0, 1}.

    The ghost a network makes outside its training poses is faint, while a real garment is fully opaque except
    for a thin soft edge. With camera noise parts of the ghost reach alpha 0.5-0.9 and join the garment, so the
    core is cut at a high threshold. Confident islands that do not touch the body, or that are tiny next to the
    garment (less than `min_share` of the largest region), are dropped. Unlike keeping just the largest region,
    this keeps a sleeve that a raised arm separates from the torso.
    """
    core = (alpha[0, 0] > threshold).to(torch.uint8).cpu().numpy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n > 1:
        body = (roi_vm[0].amax(0) > 0).cpu().numpy()
        keep = np.zeros(n, bool)
        keep[np.unique(labels[body & (core > 0)])] = True
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        areas[0] = 0
        keep &= areas >= min_share * areas.max()
        keep[0] = False
        core = keep[labels].astype(np.uint8)
    k = 2 * grow + 1
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return torch.from_numpy(core).to(alpha.device, torch.float32)[None, None]


_GAUSS = {}


def gaussian_blur(img, sigma):
    """cv2.GaussianBlur(img, (0, 0), sigma) for a float (H, W) tensor (kernel size and reflect-101 border as OpenCV)."""
    ksize = int(round(sigma * 4 * 2 + 1)) | 1  # OpenCV's automatic size for float images
    key = (ksize, sigma, img.device)
    if key not in _GAUSS:
        kernel = torch.from_numpy(cv2.getGaussianKernel(ksize, sigma).astype(np.float32)[:, 0]).to(img.device)
        _GAUSS[key] = kernel
    kernel = _GAUSS[key]
    r = ksize // 2
    x = F.pad(img[None, None], (r, r, r, r), mode='reflect')
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x[0, 0]


def compose(frame, garment, alpha, occlusion, premultiplied, alpha_low=0.1, alpha_high=0.9, global_alpha=1.0,
            blend='soft'):
    """GarmentCompositor on the GPU.

    frame: uint8 (H, W, 3); garment: float (H, W, 3) in 0..255; alpha: float (H, W) in 0..1;
    occlusion: float (H, W) in {0, 1} or None. Returns uint8 (H, W, 3) on the same device.
    blend='original' reproduces the repository's original naive_overlay_alpha: hard mask (alpha > 128/255),
    eroded by one pixel, garment colour used as is.
    """
    alpha = alpha.clamp(0.0, 1.0)
    if blend == 'original':
        hard = (torch.floor(alpha * 255.0) > 128).float()[None, None]
        a = -F.max_pool2d(-hard, 3, stride=1, padding=1)[0, 0] * global_alpha  # erosion with a 3x3 square
        if occlusion is not None:
            a = a * (1.0 - occlusion)
        a = a[..., None]
        out = garment.floor() * a + frame.float() * (1.0 - a)
        return out.round().clamp(0, 255).to(torch.uint8)
    if premultiplied:
        valid = (alpha > 1e-3)[..., None]
        garment = (garment / alpha.clamp(min=1e-3)[..., None]).clamp(0, 255) * valid
    a = ((alpha - alpha_low) / (alpha_high - alpha_low)).clamp(0.0, 1.0) * global_alpha
    if occlusion is not None:
        a = a * (1.0 - gaussian_blur(occlusion, 1.5))
    a = a[..., None]
    out = garment * a + frame.float() * (1.0 - a)
    return out.round().clamp(0, 255).to(torch.uint8)
