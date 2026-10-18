import os

import numpy as np
import cv2


def _tile(img, size, label):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return img


def compose_debug_panel(output, debug):
    """Show the try-on output next to what the garment network saw and produced for this frame.

    debug: dict from FrameProcessor with 512x512 ROI images: 'input' (camera frame), 'vm' (3D body render),
    'sdp' (DensePose body map), 'garment' (network colour output), 'alpha' (float garment mask) and
    'occlusion' (hands/face kept in front of the garment, uint8).
    """
    h = output.shape[0]
    size = h // 3
    alpha = np.clip(debug['alpha'], 0, 1)
    # square root brightens faint alpha, so weak garment "ghosts" are visible; red: protected hands/face
    alpha_view = cv2.cvtColor((np.sqrt(alpha) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    alpha_view[debug['occlusion'] > 127] = (0, 0, 255)
    pad = lambda col: np.vstack([col, np.zeros((h - col.shape[0], size, 3), np.uint8)])
    left = pad(np.vstack([_tile(debug['input'], size, 'camera input'),
                          _tile(debug['vm'], size, 'VM (BEV body)'),
                          _tile(debug['sdp'], size, 'SDP (DensePose)')]))
    right = pad(np.vstack([_tile(debug['garment'], size, 'garment colour'),
                           _tile(alpha_view, size, 'sqrt(alpha), red=kept'),
                           np.zeros((size, size, 3), np.uint8)]))
    return np.hstack([output, left, right])


def save_debug(path_png, panel, debug):
    """Save the displayed panel, plus the full-frame arrays as .npz for offline analysis."""
    os.makedirs(os.path.dirname(path_png) or '.', exist_ok=True)
    cv2.imwrite(path_png, panel)
    if debug is not None:
        np.savez_compressed(os.path.splitext(path_png)[0] + '.npz', **debug['raw'])
