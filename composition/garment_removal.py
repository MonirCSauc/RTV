import torch
import torch.nn.functional as F

# DensePose part labels:
# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
TORSO = (1, 2)
ARMS = (15, 16, 17, 18, 19, 20, 21, 22)
HANDS = (3, 4)
HEAD = (23, 24)
UPPER_LEGS = (7, 8, 9, 10)
SKIN_SOURCE = HANDS + HEAD


def _lut(indices, device):
    lut = torch.zeros(25, dtype=torch.bool, device=device)
    lut[list(indices)] = True
    return lut


def _dilate(mask, k):
    # separable, the kernels get large
    k = max(int(k) | 1, 1)
    x = F.max_pool2d(mask[None, None].float(), (k, 1), 1, (k // 2, 0))
    return F.max_pool2d(x, (1, k), 1, (0, k // 2))[0, 0] > 0.5


def _blur(img, k):
    # img: (H, W) or (H, W, C) float
    k = max(int(k) | 1, 1)
    x = img[None, None] if img.dim() == 2 else img.permute(2, 0, 1)[None]
    x = F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    return x[0, 0] if img.dim() == 2 else x[0].permute(1, 2, 0)


class WornGarmentRemover:
    """Erases the upper-body garment the person is actually wearing wherever the synthesized garment
    does not cover it, so the try-on result is not a plain overlay.

    There is no network for this: pixels inside the body silhouette are repainted with the person's own
    skin tone (shaded by the original image), loose fabric outside it is replaced by a learned background
    plate. The plate is learned from pixels the person is not standing in front of, so loose fabric can
    only be erased once the person has moved away from that spot at least once.
    """

    def __init__(self, bg_threshold=28.0, bg_rate=0.05):
        self.bg = None
        self.bg_valid = None
        self.skin = None
        self.bg_threshold = bg_threshold
        self.bg_rate = bg_rate

    def reset_background(self):
        self.bg = None
        self.bg_valid = None

    def _check_size(self, frame):
        if self.bg is None or self.bg.shape != frame.shape:
            self.bg = frame.clone()
            self.bg_valid = torch.zeros(frame.shape[:2], dtype=torch.bool, device=frame.device)

    def observe_empty(self, frame):
        # frame: (H, W, 3) float BGR with nobody in it
        self._check_size(frame)
        self.bg = torch.where(self.bg_valid[..., None], torch.lerp(self.bg, frame, self.bg_rate), frame)
        self.bg_valid[:] = True

    def _update_skin(self, frame, labels):
        src = _lut(SKIN_SOURCE, frame.device)[labels]
        if int(src.sum()) < 50:
            return
        px = frame[src]
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128
        cb = (b - y) * 0.564 + 128
        skin_like = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
        if int(skin_like.sum()) >= 50:
            px = px[skin_like]
        elif self.skin is not None:
            return
        tone = px.median(0).values
        self.skin = tone if self.skin is None else torch.lerp(self.skin, tone, 0.1)

    def __call__(self, frame, labels, body, alpha, roi_side):
        """
        frame: (H, W, 3) float BGR image
        labels: (H, W) long DensePose part labels
        body: (H, W) bool silhouette of the fitted upper-body mesh
        alpha: (H, W) float coverage of the synthesized garment
        roi_side: side length in pixels of the upper-body crop, used to scale the kernels
        returns the frame with the worn garment erased where the new one leaves it visible
        """
        self._check_size(frame)
        self._update_skin(frame, labels)
        device = frame.device
        r = max(int(roi_side * 0.12), 3)

        torso = _lut(TORSO, device)[labels]
        upper = torso | _lut(ARMS, device)[labels]
        person = (labels > 0) | body | (alpha > 0.05)
        zone = _dilate(upper | body, 2 * r + 1)

        # learn the background wherever the person is not
        outside = ~_dilate(person, 2 * r + 1)
        self.bg = torch.where((outside & self.bg_valid)[..., None], torch.lerp(self.bg, frame, self.bg_rate),
                              torch.where(outside[..., None], frame, self.bg))
        self.bg_valid |= outside

        rows = torch.arange(frame.shape[0], device=device)[:, None]
        body_rows = body.any(1).nonzero()
        if len(body_rows) == 0:
            return frame
        top, hem = int(body_rows[0]), int(body_rows[-1])

        # loose fabric: differs from the background, next to the upper body, not a labelled body part
        diff = (_blur(frame, 5) - self.bg).abs().amax(-1)
        loose = (diff > self.bg_threshold) & self.bg_valid & (labels == 0) & zone & (rows <= hem + r // 2)

        worn = _dilate(upper | loose, 5) & ~_lut((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24), device)[labels]
        # cuffs and collars get labelled as hand and head, but the mesh tells where the wrists and neckline are
        cuffs = _lut(HANDS, device)[labels] & _dilate(body, 2 * (r // 4) + 1)
        collar = _lut(HEAD, device)[labels] & (rows >= top - r // 6)
        worn = (worn | cuffs | collar) & zone
        # the uncovered neck stays skin even though the mesh starts below it
        neck = collar | (torso & (rows < top + r // 2))
        # below the new garment's hem the body is covered by the trousers, not bare
        covered = alpha > 0.5
        below_hem = _dilate(torso, 5) & (covered.cumsum(0) > 0) & (covered.flip(0).cumsum(0).flip(0) == 0)
        legs = _lut(UPPER_LEGS, device)[labels]
        has_legs = int(legs.sum()) >= 50
        legs_region = worn & below_hem & has_legs
        skin_region = worn & (body | neck | cuffs) & ~legs_region & (self.skin is not None)
        bg_region = worn & ~skin_region & ~legs_region & self.bg_valid

        fill = frame
        lum = _blur(frame.mean(-1), 9)
        for region, tone in ((skin_region, self.skin), (legs_region, frame[legs].median(0).values if has_legs else None)):
            if tone is not None and bool(region.any()):
                shade = (lum / lum[region].median().clamp(min=1.0)).clamp(0.75, 1.15)
                fill = torch.where(region[..., None], tone * shade[..., None], fill)
        fill = torch.where(bg_region[..., None], self.bg, fill)

        weight = _blur((skin_region | legs_region | bg_region).float(), 5)[..., None]
        return torch.lerp(frame, fill, weight)
