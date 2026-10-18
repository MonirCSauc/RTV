import numpy as np
import cv2


def _tile(img, size, label):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def compose_debug_panel(output, debug):
    """Show the try-on output next to what the garment network saw for this frame.

    debug: dict from FrameProcessor with 512x512 ROI images 'vm' (3D body render), 'sdp' (DensePose body map),
    'alpha' (float garment mask) and 'occlusion' (hands/face kept in front, uint8).
    """
    h = output.shape[0]
    size = h // 3
    alpha = (np.clip(debug['alpha'], 0, 1) * 255).astype(np.uint8)
    alpha_view = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    alpha_view[debug['occlusion'] > 127] = (0, 0, 255)  # red: hands/face protected from the garment
    column = np.vstack([_tile(debug['vm'], size, 'VM (BEV body)'),
                        _tile(debug['sdp'], size, 'SDP (DensePose)'),
                        _tile(alpha_view, size, 'alpha + protected')])
    column = np.vstack([column, np.zeros((h - column.shape[0], size, 3), np.uint8)])
    return np.hstack([output, column])
