import numpy as np
import cv2


def unpremultiply(rgb, alpha, eps=1e-3):
    # rgb: HxWx3 float, already multiplied by alpha (HxW float in [0,1])
    a = np.maximum(alpha, eps)[..., None]
    return np.clip(rgb / a, 0, 255) * (alpha[..., None] > eps)


def fill_from_interior(rgb, mask, band=16, levels=5, erode_px=2, downscale=4):
    """Replace the colours at and around a garment edge with colours pulled from its interior.

    Garment crops are cut out on a black background and then warped/resized with bilinear
    interpolation, so pixels along the edge are blended with black. Training on them teaches
    the network to draw a dark outline. Only pixels well inside the mask are trusted; the edge
    and a band outside it are filled by push-pull interpolation of the trusted colours.

    rgb: HxWx3 uint8, mask: HxW uint8 (0..255). Returns HxWx3 uint8.
    """
    trusted = (mask > 250).astype(np.uint8)
    if erode_px > 0:
        trusted = cv2.erode(trusted, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    w = trusted.astype(np.float32)
    if w.sum() == 0:
        return rgb.copy()
    c = rgb.astype(np.float32) * w[..., None]
    # the fill only feeds the thin edge band, so it is computed at reduced resolution for speed
    h, wd = mask.shape[:2]
    small = (max(1, wd // downscale), max(1, h // downscale))
    c = cv2.resize(c, small, interpolation=cv2.INTER_AREA)
    w = cv2.resize(w, small, interpolation=cv2.INTER_AREA)

    # push: build a pyramid of weighted colour sums
    pyr = [(c, w)]
    for _ in range(levels):
        c, w = pyr[-1]
        pyr.append((cv2.pyrDown(c), cv2.pyrDown(w)))
    # pull: fill each level's holes from the coarser level
    c, w = pyr[-1]
    filled = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(pyr[:-1]):
        up = cv2.pyrUp(filled, dstsize=(w.shape[1], w.shape[0]))
        wc = np.clip(w, 0, 1)[..., None]
        filled = c / np.maximum(w, 1e-6)[..., None] * wc + up * (1 - wc)
    filled = cv2.resize(filled, (wd, h), interpolation=cv2.INTER_LINEAR)

    out = rgb.astype(np.float32)
    keep = trusted.astype(bool)
    near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8)).astype(bool)
    replace = near & ~keep
    out[replace] = filled[replace]
    out[~near] = 0
    return np.clip(out, 0, 255).astype(np.uint8)
