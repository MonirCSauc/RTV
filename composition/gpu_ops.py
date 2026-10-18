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


def warp_affine(src, M, out_h, out_w):
    """Like cv2.warpAffine(src, M, (out_w, out_h)): src (N, C, H, W) float, M (2x3) maps src to output pixels."""
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
    return F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


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


_ELLIPSE = {}


def hand_head_mask(iuv, dilate_px=2):
    """IUV2HandHeadMask on the GPU: float (H, W) in {0, 1}, dilated with OpenCV's elliptical kernel."""
    label = iuv[..., 0]
    mask = ((label == 3) | (label == 4) | (label == 23) | (label == 24)).float()[None, None]
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        if (k, iuv.device) not in _ELLIPSE:
            kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32))
            _ELLIPSE[(k, iuv.device)] = kernel.to(iuv.device)[None, None]
        mask = (F.conv2d(mask, _ELLIPSE[(k, iuv.device)], padding=dilate_px) > 0).float()
    return mask[0, 0]


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


def compose(frame, garment, alpha, occlusion, premultiplied, alpha_low=0.1, alpha_high=0.9, global_alpha=1.0):
    """GarmentCompositor on the GPU.

    frame: uint8 (H, W, 3); garment: float (H, W, 3) in 0..255; alpha: float (H, W) in 0..1;
    occlusion: float (H, W) in {0, 1} or None. Returns uint8 (H, W, 3) on the same device.
    """
    alpha = alpha.clamp(0.0, 1.0)
    if premultiplied:
        valid = (alpha > 1e-3)[..., None]
        garment = (garment / alpha.clamp(min=1e-3)[..., None]).clamp(0, 255) * valid
    a = ((alpha - alpha_low) / (alpha_high - alpha_low)).clamp(0.0, 1.0) * global_alpha
    if occlusion is not None:
        a = a * (1.0 - gaussian_blur(occlusion, 1.5))
    a = a[..., None]
    out = garment * a + frame.float() * (1.0 - a)
    return out.round().clamp(0, 255).to(torch.uint8)
