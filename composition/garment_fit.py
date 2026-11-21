"""Make the synthesized garment cover the customer's own top where it would show around it.

The method draws the new garment over the camera frame and does not remove what the customer wears (paper,
section 5). When their own top is looser, longer-sleeved or higher-necked than the new garment, it shows as a
strip on the shoulders, at the sleeve ends, at the sides and at the neck. Nothing is painted onto the customer
here. Instead the new garment is changed in three ways, all before it is blended:

- it is grown over the adjacent pixels that show the customer's own top (recognised by the colour of the cloth
  the new garment already covers), using the garment's own colour next to them;
- small holes in it are closed;
- the head and hand masks, which keep those parts in front of the garment, give way where they hold the
  customer's own top instead of skin (DensePose counts the collar as neck and the cloth around a fist as hand).

It needs no training and works from what the pipeline already has: the DensePose part map, the camera frame and
the garment with its alpha.

Limits: the grown part has the garment's local colour but not its texture, so it is kept short (`REACH`); a top
that is much larger than the new garment still shows beyond that, and so does one that is longer (nothing is
done below the hem). Where the background has the colour of the customer's top, the garment is not grown beyond
the body DensePose labels, because the top cannot be told from the background there.
"""
import torch
import torch.nn.functional as F

TORSO = (1, 2)
UPPER_ARMS = (15, 16, 17, 18)
LOWER_ARMS = (19, 20, 21, 22)
HANDS = (3, 4)
HEAD = (23, 24)


def _labels(label, parts):
    mask = torch.zeros_like(label, dtype=torch.bool)
    for p in parts:
        mask |= label == p
    return mask


def _dilate(mask, radius):
    # mask: bool or float (H, W); square dilation, separable; returns bool
    if radius <= 0:
        return mask.bool()
    k = 2 * radius + 1
    x = mask.float()[None, None]
    x = F.max_pool2d(x, (1, k), stride=1, padding=(0, radius))
    x = F.max_pool2d(x, (k, 1), stride=1, padding=(radius, 0))
    return x[0, 0] > 0.5


def _erode(mask, radius):
    return ~_dilate(~mask.bool(), radius)


def _box(x, radius):
    # box filter of a (C, H, W) tensor, separable, zero outside the image
    k = 2 * radius + 1
    x = F.avg_pool2d(x[None], (1, k), stride=1, padding=(0, radius))
    return F.avg_pool2d(x, (k, 1), stride=1, padding=(radius, 0))[0]


def _chroma(bgr):
    # Cr, Cb of a float BGR image (0..255): skin of any tone lies in a compact range of these two
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (r - y) * 0.713 + 128.0, (b - y) * 0.564 + 128.0


class GarmentFitter:
    SKIN_CR = (133.0, 180.0)   # generic skin range, used only to pick skin samples from hands, arms and face
    SKIN_CB = (80.0, 135.0)
    MIN_SKIN_SAMPLES = 150
    CLOTH_TOLERANCE = 70.0     # largest colour difference (0..255, any channel) to the customer's top
    MIN_CONTRAST = 50.0        # smallest difference between top and skin for the two to be told apart
    REACH = 0.14               # how far the garment may grow, as a fraction of its height
    BESIDE = 0.10              # how far from the labelled body the customer's top is looked for, same unit
    MAX_BACKGROUND_SHARE = 0.25  # of the background near a pixel that may have the top's colour for it to grow there
    FILL_RADII = (2, 5, 12, 28)

    def __init__(self):
        self.reset()

    def reset(self):
        self.skin = None       # (bgr colour, Cr, Cb) of the customer, smoothed over time
        self.last_grown = None

    forget_person = reset

    def _update_skin(self, frame, label, cr, cb):
        samples = _labels(label, HANDS + LOWER_ARMS + HEAD)
        samples &= (cr > self.SKIN_CR[0]) & (cr < self.SKIN_CR[1]) & (cb > self.SKIN_CB[0]) & (cb < self.SKIN_CB[1])
        if int(samples.sum()) >= self.MIN_SKIN_SAMPLES:
            new = (frame[samples].median(0).values, cr[samples].median(), cb[samples].median())
            self.skin = new if self.skin is None else tuple(o + 0.2 * (n - o) for o, n in zip(self.skin, new))

    def _fill_colour(self, garment, solid, region):
        """Garment colour continued outwards: for every pixel of `region`, the average garment colour in the
        smallest neighbourhood that holds enough of the solid garment."""
        ys, xs = torch.nonzero(region, as_tuple=True)
        pad = self.FILL_RADII[-1] + 1
        h, w = region.shape
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
        weight = solid[y0:y1, x0:x1].float()[None]
        colour = garment[y0:y1, x0:x1].permute(2, 0, 1) * weight
        fill = torch.zeros_like(colour)
        found = torch.zeros_like(weight, dtype=torch.bool)
        for r in self.FILL_RADII:
            wt = _box(weight, r)
            ok = (wt > 0.08) & ~found
            fill = torch.where(ok, _box(colour, r) / wt.clamp(min=1e-4), fill)
            found |= ok
        out = torch.zeros_like(garment)
        out[y0:y1, x0:x1] = fill.permute(1, 2, 0)
        reached = torch.zeros_like(region)
        reached[y0:y1, x0:x1] = found[0]
        return out, reached

    def __call__(self, frame, iuv, garment, alpha, occlusion=None):
        """frame: uint8 (H, W, 3) BGR tensor; iuv: uint8 (H, W, 3) DensePose map; garment: float (H, W, 3)
        garment colour (not premultiplied); alpha: float (H, W) in 0..1, as it will be blended;
        occlusion: float (H, W), hands/head kept in front. Returns garment, alpha and occlusion, changed."""
        frame = frame.float()
        label = iuv[..., 0]
        person = label > 0
        covered = alpha > 0.5
        cr, cb = _chroma(frame)
        self._update_skin(frame, label, cr, cb)
        self.last_grown = None
        rows = torch.nonzero(covered.any(1))
        if rows.numel() == 0:
            return garment, alpha, occlusion
        top, bottom = int(rows[0]), int(rows[-1])
        g_height = max(bottom - top, 1)
        h, w = alpha.shape
        ys = torch.arange(h, device=alpha.device)[:, None].expand(h, w)

        torso, upper_arm, lower_arm, head = (_labels(label, p) for p in (TORSO, UPPER_ARMS, LOWER_ARMS, HEAD))

        # the colour of the customer's own top, read from the camera pixels the new garment covers
        under = (alpha > 0.9) & torso
        if int(under.sum()) < 500:
            return garment, alpha, occlusion
        cloth = frame[under][::7].median(0).values
        # a pixel shows the customer's top if its colour is close to that cloth and closer to it than to their
        # skin. (Hue does not separate the two: dark cloth in warm light has the hue of skin.)
        to_cloth = (frame - cloth).abs().amax(-1)
        if self.skin is not None:
            if float((self.skin[0] - cloth).abs().amax()) < self.MIN_CONTRAST:
                return garment, alpha, occlusion   # top and skin look alike: nothing can be told apart
            skin_like = (frame - self.skin[0]).abs().amax(-1) <= to_cloth
        else:
            skin_like = torch.zeros_like(person)
        own_top = (to_cloth < self.CLOTH_TOLERANCE) & ~skin_like

        # the hem, column by column (it is curved or tilted): nothing at or below it is touched, so trousers
        # and a longer top are left alone
        # (looked for in the lower part of the garment only: higher up, the lowest garment pixel is a sleeve's)
        low = covered & (ys > top + int(0.6 * g_height))
        hem = torch.where(low, ys, torch.full_like(ys, -1)).amax(0)
        hem = torch.where(hem >= 0, hem, torch.full_like(hem, bottom)) - int(0.05 * g_height)
        above_hem = ys < hem[None, :]

        near = _dilate(covered, max(int(self.REACH * g_height), 6))
        # own top on the clothed body parts ...
        grow = (torso & above_hem | upper_arm | lower_arm) & own_top & near
        # ... and standing off the body, where DensePose labels nothing: a loose top is wider than the body
        # under it. Only close to the body, so that a background of the same colour is not taken for the top.
        head_rows = torch.nonzero(head.any(1))
        h0, h1 = (int(head_rows[0]), int(head_rows[-1])) if head_rows.numel() > 0 else (0, top)
        neck_zone = head & (ys > h1 - int(0.3 * (h1 - h0)))
        beside_r = max(int(self.BESIDE * g_height), 4)
        zone = _dilate(torso | upper_arm, beside_r)
        beside = zone & ~person
        # ... and only where the background is not of the top's colour (a white top in front of a light wall):
        # there the wall next to the body would be taken for the top and the garment painted onto it. The
        # background is read from a ring just beyond the zone; where it cannot be read, nothing is grown.
        ring = _dilate(zone, max(beside_r // 2, 4)) & ~zone & ~person
        reach = 2 * beside_r
        like_top = ring & (to_cloth < self.CLOTH_TOLERANCE + 20.0)
        seen = _box(ring.float()[None], reach)[0]
        beside &= (seen > 1e-3) & (_box(like_top.float()[None], reach)[0] < self.MAX_BACKGROUND_SHARE * seen)
        # (bare skin is always labelled, so the colour need not be told from skin here)
        grow |= (beside & (to_cloth < self.CLOTH_TOLERANCE + 20.0) & near & above_hem &
                 (ys > h1 - int(0.05 * (h1 - h0))))   # below the head: dark hair is not the top
        # the collar in the neck opening, which DensePose counts as the lowest part of the head: only what is
        # very close to the top's colour (the shadow under the chin is dark too) and has skin above it (seen
        # from behind, dark hair at the nape does not)
        skin_above = (skin_like & head).float().cumsum(0) >= 3
        grow |= (neck_zone & (to_cloth < 0.5 * self.CLOTH_TOLERANCE) & ~skin_like & skin_above &
                 _dilate(covered, max(int(0.03 * g_height), 3)))
        # holes in the garment itself
        r = max(int(0.04 * g_height), 3)
        grow |= _erode(_dilate(covered, r), r) & (torso | upper_arm) & ~neck_zone
        grow &= ~covered

        # only what touches the garment, directly or through other such pixels (not a dark object behind)
        step = max(int(0.02 * g_height), 2)
        linked = _dilate(covered, 2) & grow
        for _ in range(max(int(self.REACH * g_height) // step, 1)):
            linked = _dilate(linked, step) & grow
        # one pixel more: the outline of the top is a mix of its colour and the background's
        # (and closed along the outline, so that a broken row of such pixels does not make a ragged edge)
        grow = _erode(_dilate(linked, 3), 2) & ~covered & ~(head & ~neck_zone) & ~(torso & ~above_hem)
        grow &= ~(lower_arm | _labels(label, HANDS)) | linked
        # keep what makes a smooth outline together with the garment; single pixels sticking out are dropped
        grow &= _box((grow | covered).float()[None], 2)[0] > 0.5

        if occlusion is not None:
            # head: in its lowest part only skin stays in front; hands: the cloth around them does not
            hands = (occlusion > 0.5) & ~_dilate(head, 4)
            give_way = (neck_zone & ~skin_like & covered) | (hands & own_top)
            occlusion = occlusion * (~_dilate(give_way, 1)).float()
        if not bool(grow.any()):
            return garment, alpha, occlusion

        solid = _erode(alpha > 0.9, 2)   # away from the garment's edge, whose colour is mixed with nothing
        region = _dilate(grow, 2)
        fill, reached = self._fill_colour(garment, solid, region)
        grow &= reached
        # the garment's own edge next to the grown part is half transparent and would show the top through it
        solid_now = grow | (region & reached & (alpha > 0.2))
        a = ((alpha - 0.5) * 2.0).clamp(0, 1)[..., None]   # the colour of a nearly transparent pixel is noise
        mixed = garment * a + fill * (1 - a)   # where the garment is thin or absent: the continued colour
        garment = torch.where((region & reached)[..., None], mixed, garment)
        alpha = torch.maximum(alpha, torch.maximum(_box(solid_now.float()[None], 1)[0], solid_now.float()))
        self.last_grown = grow
        return garment, alpha, occlusion
