"""Customer tracking and body-motion smoothing on top of BEV's per-frame estimates.

BEV estimates every person in every frame independently, so (1) "the nearest person" can switch to a bystander
from one frame to the next and (2) the estimated pose, position and shape tremble even when the customer stands
still, which makes the rendered body, the crop and finally the garment tremble.

CustomerTracker keeps the garment on one person. BodySmoother filters that person's pose, position and shape
over time with One-Euro filters (strong smoothing at rest, almost none in fast motion).
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from util.one_euro import OneEuroFilter

# SMPL joints: 0 pelvis, 9 upper spine, 12 neck, 15 head
_TORSO = [0, 9, 12]


def person_features(outputs):
    """Per person: torso centre (N, 2) in image pixels, head-to-pelvis length (N,) in pixels, depth (N,)."""
    joints = outputs['pj2d_org'][:, :24].detach().cpu().numpy()
    centre = joints[:, _TORSO].mean(1)
    size = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=1) + 1e-6
    depth = outputs['cam_trans'][:, 2].detach().cpu().numpy()
    return centre, size, depth


def select_person(outputs, index):
    """Keep only person `index` in a BEV output dict (every per-person tensor keeps a leading dimension of 1)."""
    n = outputs['cam_trans'].shape[0]
    selected = {}
    for key, value in outputs.items():
        if key != 'smpl_face' and hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == n:
            selected[key] = value[index:index + 1]
        else:
            selected[key] = value
    return selected


class CustomerTracker:
    """Chooses one person and stays with them.

    Acquire: among people whose torso centre is inside the mirror zone (a horizontal band of the image) and who
    are large enough, take the nearest (smallest depth), as the original code did every frame.
    Follow: afterwards take the person closest to where the customer was in the previous frame, provided their
    apparent size is about the same. Someone else who walks by, even in front of the customer and closer to the
    camera, does not take the garment: they are at the wrong place or the wrong size. While the customer is hidden
    no garment is drawn; if they stay hidden for `max_lost` seconds, the tracker lets go and acquires again.
    """

    def __init__(self, zone=(0.0, 1.0), min_size=0.0, gate=0.5, max_lost=1.5, size_change=1.25):
        self.zone = zone                # (left, right) as fractions of the image width
        self.min_size = min_size        # minimum head-to-pelvis length as a fraction of the image height
        self.gate = gate                # how far (in body lengths) the customer may move between two frames
        self.max_lost = max_lost        # seconds
        self.size_change = size_change  # largest plausible change of apparent size between two sightings
        self.reset()

    def reset(self):
        self.centre = None
        self.size = None
        self.lost_since = None      # time the customer was last seen, while they are missing
        self.t = None
        self.new_person = False     # True on the frame a (new) customer was acquired

    def _lost(self, t):
        """Customer not in this frame. Returns True once they have been gone too long (the tracker lets go)."""
        if self.lost_since is None:
            self.lost_since = self.t if self.t is not None else t
        if t - self.lost_since >= self.max_lost:
            self.reset()
            return True
        return False

    def miss(self, t):
        """Call when BEV found nobody in this frame."""
        self.new_person = False
        if self.centre is not None:
            self._lost(t)

    def __call__(self, outputs, img_shape, t):
        """Returns the index of the customer in `outputs`, or None if they are not in this frame."""
        centre, size, depth = person_features(outputs)
        self.new_person = False
        if self.centre is not None:
            dist = np.linalg.norm(centre - self.centre, axis=1) / self.size
            ratio = np.maximum(size / self.size, self.size / size)
            # the allowed step and size change grow a little while the customer is missing (they keep moving)
            missing = 0.0 if self.lost_since is None else t - self.lost_since
            ok = (dist <= self.gate * (1.0 + missing)) & (ratio <= self.size_change * (1.0 + 0.2 * missing))
            if ok.any():
                best = int(np.argmin(np.where(ok, dist, np.inf)))
                self.centre, self.size, self.lost_since, self.t = centre[best], size[best], None, t
                return best
            if not self._lost(t):
                return None
        h, w = img_shape[:2]
        ok = ((centre[:, 0] >= self.zone[0] * w) & (centre[:, 0] <= self.zone[1] * w) &
              (size >= self.min_size * h))
        if not ok.any():
            return None
        best = int(np.argmin(np.where(ok, depth, np.inf)))
        self.centre, self.size, self.lost_since, self.t = centre[best], size[best], None, t
        self.new_person = True
        return best


class NearestPerson:
    """The original choice: the person nearest to the camera, decided anew in every frame."""
    new_person = False

    def reset(self):
        pass

    def miss(self, t):
        pass

    def __call__(self, outputs, img_shape, t):
        return int(torch.argmin(outputs['cam_trans'][:, 2]).item())


class BodySmoother:
    """Smooths one person's SMPL parameters over time and rebuilds the mesh and its projection from them.

    - Joint rotations: One-Euro filter on unit quaternions (one cutoff per joint).
    - Camera parameters (scale and image position): One-Euro filter.
    - Body shape: a person's shape does not change, so it is a running average over the first seconds and a very
      slow average afterwards. This removes the "breathing" of the garment's width.
    """

    def __init__(self, pose_cutoff=1.5, pose_beta=4.0, cam_cutoff=1.2, cam_beta=8.0, shape_seconds=3.0):
        self.pose = OneEuroFilter(pose_cutoff, pose_beta, group_axis=-1)
        self.cam = OneEuroFilter(cam_cutoff, cam_beta)
        self.shape_seconds = shape_seconds
        self.reset()

    def reset(self):
        self.pose.reset()
        self.cam.reset()
        self.betas = None
        self.shape_time = 0.0
        self.prev_quat = None
        self.t = None

    def __call__(self, outputs, t, bev_model):
        """outputs: BEV dict holding one person; t: frame time in seconds; bev_model: MyBEV (rebuilds the mesh)."""
        dt = 0.0 if self.t is None else t - self.t
        if dt > 1.0:  # a long gap: the old state says nothing about now
            self.reset()
            dt = 0.0
        self.t = t
        device, dtype = outputs['smpl_thetas'].device, outputs['smpl_thetas'].dtype

        thetas = outputs['smpl_thetas'][0].detach().cpu().numpy().reshape(24, 3)
        quat = Rotation.from_rotvec(thetas).as_quat()
        if self.prev_quat is not None:  # q and -q are the same rotation: stay on the previous side
            quat[np.sum(quat * self.prev_quat, axis=1) < 0] *= -1
        quat = self.pose(quat, dt).astype(np.float64)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        self.prev_quat = quat
        thetas = Rotation.from_quat(quat).as_rotvec().reshape(1, 72)

        cam = self.cam(outputs['cam'][0].detach().cpu().numpy(), dt)[None]

        betas = outputs['smpl_betas'][0].detach().cpu().numpy().astype(np.float64)
        if self.betas is None:
            self.betas, self.shape_time = betas, 0.0
        else:
            self.shape_time += dt
            # running mean at first, then an average over about shape_seconds * 4
            a = dt / max(min(self.shape_time, self.shape_seconds * 4.0), dt)
            self.betas = self.betas + a * (betas - self.betas)

        smoothed = dict(outputs)
        smoothed['smpl_thetas'] = torch.as_tensor(thetas, dtype=dtype, device=device)
        smoothed['smpl_betas'] = torch.as_tensor(self.betas[None], dtype=dtype, device=device)
        smoothed['cam'] = torch.as_tensor(cam, dtype=dtype, device=device)
        smoothed.update(bev_model.rebuild(smoothed))
        return smoothed
