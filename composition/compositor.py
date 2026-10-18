import numpy as np
import cv2

from util.matte_util import unpremultiply


class GarmentCompositor:
    """Blends a synthesized garment (colour + alpha) over the camera frame.

    premultiplied: the garment colour was learned already multiplied by alpha, i.e. darkened
        along the edge. True for checkpoints trained on black-background crops (all released
        rtv_ckpts); False for checkpoints trained with straight colour (see rtv_meta.json).
    alpha_low/alpha_high: alpha below alpha_low is dropped (removes faint haze around the garment),
        alpha above alpha_high is fully opaque; in between stays soft, giving anti-aliased edges.
    """

    def __init__(self, premultiplied=True, alpha_low=0.1, alpha_high=0.9):
        self.premultiplied = premultiplied
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

    def __call__(self, frame, garment, alpha, occlusion_mask=None):
        # frame, garment: HxWx3 uint8 (same space); alpha: HxW float [0,1] or uint8 [0,255]
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        garment = garment.astype(np.float32)
        if self.premultiplied:
            garment = unpremultiply(garment, alpha)
        a = np.clip((alpha - self.alpha_low) / (self.alpha_high - self.alpha_low), 0.0, 1.0)
        if occlusion_mask is not None:
            # hands and face stay in front; the soft edge sits on their boundary
            occ = cv2.GaussianBlur(occlusion_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5)
            a = a * (1.0 - occ)
        a = a[..., None]
        out = garment * a + frame.astype(np.float32) * (1.0 - a)
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
