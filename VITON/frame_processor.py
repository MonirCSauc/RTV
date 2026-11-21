"""One per-frame try-on pipeline for every garment network in the repository.

Two kinds of garment network exist (see TUTORING.md 1.4):

- 'upper':    the earlier per-frame method [57]. pix2pixHD, 512x512 crop of the upper body, input = upper-body VM
              render + DensePose map with UV (SDP). Checkpoints: lab_*, jin_* (`*_vmsdp2ta`).
- 'full_seq': the paper's ReGarSyn. pix2pixHD + ConvLSTM, 576x432 crop of the whole body, input = full-body VM
              render + simplified DensePose map without UV (SSDP). Checkpoints: `*_seq_vmssdp2ta_576`.

Both share every other step: BEV body fit, customer tracking and smoothing, DensePose, the garment network on the
GPU, and compositing. A garment's kind is read from `rtv_meta.json` ('kind') or else from its name.
"""
import threading
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from SMPL.smpl_regressor import SMPL_Regressor
from model.DensePose.densepose_extractor import DensePoseExtractor, BoxFilter
from model.garment_model import make_pix2pix_model, read_meta
from composition.compositor import GarmentCompositor
from composition import gpu_ops
from composition.garment_fit import GarmentFitter
from util.cv2_trans_util import TemporalSmoothing, get_inverse_trans
from util.one_euro import OneEuroFilter
from util.latency import NULL_TIMING
from util.gpu_check import report_devices


class GarmentKind:
    """Everything that differs between the two kinds of garment network."""

    def __init__(self, name, roi_h, roi_w, model_name, recurrent, body_shape, alpha_gate, warp_mode,
                 roi_ema, roi_cutoff, roi_beta, temporal):
        self.name = name
        self.roi_h, self.roi_w = roi_h, roi_w
        self.model_name = model_name
        self.recurrent = recurrent      # the network keeps a memory of earlier frames (ConvLSTM)
        self.body_shape = body_shape    # default --body_shape
        self.alpha_gate = alpha_gate    # whether the body gate applies (loose garments reach far beyond the body)
        self.warp_mode = warp_mode      # how the network inputs are resampled into the crop
        self.roi_ema = roi_ema          # crop smoothing of the original code (per frame), used with --smooth off
        self.roi_cutoff, self.roi_beta = roi_cutoff, roi_beta  # crop smoothing in Hz (frame-rate independent)
        self.temporal = temporal        # whether the output flicker filter applies (not needed with a ConvLSTM)
        self._template = None

    def template(self):
        # the VM body template; created on first use because it opens an OpenGL context
        if self._template is None:
            if self.name == 'upper':
                from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
                self._template = UpperBodySMPL()
            else:
                from SMPL.fullbody_smpl.FullBody import FullBodySMPL
                self._template = FullBodySMPL()
        return self._template

    def crop(self, regressor, smpl_param):
        """Affine transform from the camera frame to the network's crop, from the body joints."""
        if self.name == 'upper':
            joints = SMPL_Regressor.chosen_joints(smpl_param, n_joints=71)
            return regressor.get_trans2roi(joints, s=1.45, img_size=self.roi_h)[0]
        return regressor.get_fullbody_trans2roi(smpl_param, s=1.4, new_h=self.roi_h, new_w=self.roi_w)[0]

    def body_map(self, iuv):
        return gpu_ops.iuv_to_sdp(iuv) if self.name == 'upper' else gpu_ops.iuv_to_ssdp(iuv)

    def alpha(self, net_out):
        # the upper-body networks were trained with the mask in 0..1, the recurrent ones with it in -1..1
        a = net_out[:, 3:4]
        return a.clamp(0.0, 1.0) if self.name == 'upper' else a.clamp(-1.0, 1.0) / 2.0 + 0.5


KINDS = {
    'upper': GarmentKind('upper', 512, 512, 'pix2pixHD', False, 'estimated', True, 'bilinear',
                         roi_ema=0.6, roi_cutoff=0.8, roi_beta=1.0, temporal=True),
    'full_seq': GarmentKind('full_seq', 576, 432, 'pix2pixHD_RNN_RGBA', True, 'train', False, 'nearest',
                            roi_ema=0.9, roi_cutoff=0.15, roi_beta=0.3, temporal=False),
}


def garment_kind(ckpt_dir, name):
    kind = read_meta(ckpt_dir, name).get('kind')
    if kind is None:
        kind = 'full_seq' if '_seq_' in name else 'upper'
    return KINDS[kind]


class RoiFilter:
    """Smooths the crop window over time, in seconds rather than frames, so it behaves the same at any frame rate.

    The crop only frames the network input (the output is warped back with the inverse), so smoothing it does not
    misalign the garment; it removes the flicker a trembling crop causes in the network's output.
    """

    def __init__(self, kind):
        self.kind = kind
        self.filter = OneEuroFilter(kind.roi_cutoff, kind.roi_beta)
        self.t = None
        self.held = None

    DEADBAND_PX = 1.0
    DEADBAND_SCALE = 0.005

    def reset(self):
        self.filter.reset()
        self.t = None
        self.held = None

    def __call__(self, trans, t, img_h):
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        sx, sy = trans[0, 0], trans[1, 1]
        w, h = self.kind.roi_w, self.kind.roi_h
        cx = ((w - 1) / 2.0 - trans[0, 2]) / sx   # crop centre in the camera frame
        cy = ((h - 1) / 2.0 - trans[1, 2]) / sy
        cx, cy, log_s = self.filter(np.array([cx / img_h, cy / img_h, np.log(sx)]), dt)
        cx, cy = cx * img_h, cy * img_h
        # hold the crop exactly still until it has drifted by a noticeable amount (a pixel, or 0.5% in size): for
        # a person standing still the network then sees the body at a constant place
        if self.held is not None and abs(cx - self.held[0]) < self.DEADBAND_PX and \
                abs(cy - self.held[1]) < self.DEADBAND_PX and abs(log_s - self.held[2]) < self.DEADBAND_SCALE:
            cx, cy, log_s = self.held
        else:
            self.held = (cx, cy, log_s)
        sx_f = float(np.exp(log_s))
        sy_f = sx_f * sy / sx
        out = np.array([[sx_f, 0.0, (w - 1) / 2.0 - sx_f * cx], [0.0, sy_f, (h - 1) / 2.0 - sy_f * cy]])
        return out, get_inverse_trans(out)


class _Garment:
    """One garment in the list: its kind, how to composite it and, once loaded, its network."""

    def __init__(self, name, ckpt_dir):
        self.name = name
        self.kind = garment_kind(ckpt_dir, name)
        self.compositor = GarmentCompositor(premultiplied=read_meta(ckpt_dir, name)['color'] == 'premultiplied')
        self.model = None


class FrameProcessor:
    # --body_shape: 'estimated' keeps BEV's estimate of the customer's shape, so the garment follows their body;
    # 'train' zeroes the shape parameters as the training data did and 'zero' zeroes all of them (paper) -- both
    # give every customer an average body. Default: 'estimated' for upper-body garments, 'train' for full-body ones.
    BODY_SHAPES = {'train': True, 'zero': 'all', 'estimated': False}

    def __init__(self, garment_name_list, ckpt_dir=None, dp_score=0.5, body_shape=None,
                 keep_garments_on_gpu=False, keep_in_front='hands_head', blend='soft', alpha_gate=30,
                 track=True, smooth=True, fp16=True, dp_mode='box', temporal=0.5, zone=(0.0, 1.0), parallel=True,
                 cover_worn=True):
        """
        track:    stay with one customer (SMPL/body_tracker.py) instead of the nearest person in each frame
        smooth:   filter the body estimate and the crop over time
        fp16:     run DensePose and the garment network in half precision
        dp_mode:  'box' = DensePose only inside the body box BEV found (about twice as fast, cannot pick up another
                  person); 'full' = the whole detector, as originally
        temporal: strength 0..1 of the output flicker filter for per-frame (upper-body) networks; 0 = off
        parallel: compute DensePose's image features while BEV runs (with dp_mode 'box')
        cover_worn: let the garment cover the customer's own top where it would show around it
        zone:     (left, right) fractions of the image width in which a new customer is accepted
        """
        self.ckpt_dir = ckpt_dir if ckpt_dir is not None else './rtv_ckpts'
        self.track, self.smooth, self.dp_mode, self.temporal = track, smooth, dp_mode, temporal
        self.fp16 = fp16 and torch.cuda.is_available()
        self.body_shape = body_shape
        self.smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=False, track=track, smooth=smooth, zone=zone)
        # detectron2 keeps detections down to a 0.05 score by default, so posters, mannequins and reflections compete
        # for "the person"; dp_score drops weak detections
        self.densepose_extractor = DensePoseExtractor(opts=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', dp_score])
        self.garment_name_list = garment_name_list
        self.garments = [_Garment(name, self.ckpt_dir) for name in garment_name_list]
        self.active = None         # the selected _Garment (None = show the camera frame)
        # Misses (no person, or DensePose disagrees with BEV): show the last garment, fully opaque, for HOLD frames so
        # a single dropout does not blink, then show the camera frame. No fading: a semi-transparent garment looks
        # worse than a brief gap.
        self.misses = 0
        self.last_good = None      # (garment, alpha, occlusion, compositor) of the last successful frame
        self.prev_trans = None     # unsmoothed crop transform of the last accepted frame, to detect a different person
        self.jump_frames = 0       # consecutive frames whose crop jumped away from prev_trans
        self.roi_filters = {}      # kind name -> crop smoothing
        self.prev_out = None       # previous network output, for the flicker filter
        self.prev_out_t = None
        self.pool = ThreadPoolExecutor(1) if (parallel and dp_mode == 'box') else None
        self.fit = GarmentFitter() if cover_worn else None
        self.box_filter = BoxFilter()  # steadies the DensePose box (with smooth and dp_mode 'box')
        self.lock = threading.Lock()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.debug = False       # when True, keep this frame's network inputs/outputs in self.last_debug
        self.last_debug = None
        self._pristine = None
        print('Model devices:')
        report_devices({'BEV': self.smpl_regressor.regressor_model,
                        'DensePose': self.densepose_extractor.predictor.model})
        print('Pipeline: tracking %s, smoothing %s, DensePose mode %s, half precision %s' % (
            'on' if track else 'off', 'on' if smooth else 'off', dp_mode, 'on' if self.fp16 else 'off'))

        self.resident = False
        self.keep_garments_on_gpu = keep_garments_on_gpu
        # body parts drawn in front of the garment; from behind, DensePose's head is the back of the head and hair,
        # which can cut into a garment's back collar
        self.keep_in_front = gpu_ops.KEEP_IN_FRONT[keep_in_front]
        self.blend = blend  # 'soft' (anti-aliased edges) or 'original' (the repository's hard 50% mask)
        self.alpha_gate = alpha_gate  # px (in the 512 crop) the garment may extend beyond the body; 0 = off
        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    # ------------------------------------------------------------------ garment networks

    # kept for callers of the earlier interface
    @property
    def viton_model(self):
        return self.active.model if self.active is not None else None

    @property
    def viton_model_list(self):
        return [g.model for g in self.garments]

    def make_model(self, garment_name):
        kind = garment_kind(self.ckpt_dir, garment_name)
        model = make_pix2pix_model(garment_name, 6, output_nc=4, model_name=kind.model_name, ckpt_dir=self.ckpt_dir)
        model.eval()
        if self.fp16:
            model.half()  # half the memory and about half the time; outputs differ by < 1/255
        return model

    GARMENT_GPU_BYTES = 800 * 2 ** 20  # one fp32 pix2pixHD garment network is ~730 MB (half that in half precision)

    def fits_on_gpu(self, n_models):
        # Keeping every garment on the GPU makes switching a pointer swap, but on Windows, running short of GPU memory
        # does not fail: the driver silently spills to system RAM and every frame slows down. So it is opt-in
        # (--keep_garments_on_gpu) and needs room for BEV, DensePose and activations on top (~2.5 GB).
        free, _ = torch.cuda.mem_get_info()
        per_model = self.GARMENT_GPU_BYTES // (2 if self.fp16 else 1)
        return free > n_models * per_model + 2.5 * 2 ** 30

    def load_all_models(self):
        resident = (self.keep_garments_on_gpu and torch.cuda.is_available() and
                    self.fits_on_gpu(len(self.garments)))
        print('Garment networks: %s' % ('all kept on the GPU (instant switching)' if resident else
                                        'kept in CPU memory; the selected one is copied to the GPU'))
        for garment in self.garments:
            try:
                new_model = self.make_model(garment.name)
            except Exception as e:  # e.g. a checkpoint folder without weights: the others must still load
                print('Could not load garment %s: %s' % (garment.name, e))
                continue
            if resident:
                new_model = new_model.to('cuda:0')
            if garment.model is None:
                garment.model = new_model
        self.resident = resident

    def load_one_models(self, garment_name):
        garment = self.garments[self.garment_name_list.index(garment_name)]
        new_model = self.make_model(garment_name)
        if garment.model is None:
            garment.model = new_model

    def switch_to_target_garment(self, garment_id):
        # load and copy to the GPU outside the lock: the frame loop keeps running on the current garment meanwhile
        garment = None
        if garment_id is not None and garment_id >= 0:
            garment = self.garments[garment_id]
            if garment.model is None:
                print("Loading from disk target garment id: ", garment_id)
                try:
                    self.load_one_models(garment.name)
                except Exception as e:
                    print('Could not load garment %s: %s' % (garment.name, e))
                    return
            garment.model.to('cuda:0')
            if garment.kind.recurrent:
                garment.model.reset()
            self._warm_up(garment)
        print("Garment %s ready on %s" % (garment.name if garment else None,
                                          next(garment.model.parameters()).device if garment else None))
        with self.lock:  # only the swap itself blocks the frame loop
            old = self.active
            self.active = garment
            self.last_good = None  # never hold the previous garment's image
            self.prev_out = None
        if old is not None and old is not garment and not self.resident:
            old.model.to('cpu')
            if old.kind.recurrent:
                old.model.reset()
            torch.cuda.empty_cache()

    def _warm_up(self, garment):
        # the first forward pass of a network is slow (cuDNN picks its algorithms); do it before the swap
        kind = garment.kind
        p = next(garment.model.parameters())
        with torch.no_grad():
            garment.model.forward(torch.zeros(1, 6, kind.roi_h, kind.roi_w, device=p.device, dtype=p.dtype))
        if kind.recurrent:
            garment.model.reset()

    def set_target_garment(self, target_id):
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    # ------------------------------------------------------------------ misses and person changes

    HOLD = 2        # missed frames that keep showing the last garment (at ~8 fps, more would visibly lag the body)
    LOST = 15       # missed frames after which the person is gone and the crop smoothing starts over
    JUMP = 0.25     # crop centre moving more than this share of the crop side, or its scale changing this much
    JUMP_FRAMES = 3 # ... in this many consecutive frames before it counts as a different person

    def _miss(self, input_frame, ft, outcome):
        # this frame produced no garment: briefly keep the last one instead of blinking
        ft.note(outcome=outcome)
        self.misses += 1
        if self.misses == self.LOST:
            self._forget_person()
        if self.last_good is not None and self.misses <= self.HOLD:
            garment, alpha, occlusion, compositor = self.last_good
            return self._compose(input_frame, garment, alpha, occlusion, compositor)
        return input_frame

    def _compose(self, frame, garment, alpha, occlusion, compositor, iuv=None):
        # blend on the GPU (composition/gpu_ops.py mirrors GarmentCompositor); one copy of the result to the CPU
        base = torch.from_numpy(frame).to(self.device)
        if self.fit is not None and iuv is not None and self.blend != 'original':
            # let the garment cover the customer's own top where it would show around it (the camera frame is
            # not changed)
            if compositor.premultiplied:
                garment = (garment / alpha.clamp(min=1e-3)[..., None]).clamp(0, 255) * (alpha > 1e-3)[..., None]
            shown = ((alpha - compositor.alpha_low) / (compositor.alpha_high - compositor.alpha_low)).clamp(0, 1)
            garment, shown, occlusion = self.fit(base, iuv, garment, shown, occlusion)
            out = gpu_ops.compose(base, garment, shown, occlusion, False, 0.0, 1.0, blend=self.blend)
        else:
            out = gpu_ops.compose(base, garment, alpha, occlusion, compositor.premultiplied,
                                  compositor.alpha_low, compositor.alpha_high, blend=self.blend)
        return out.cpu().numpy()

    def _forget_person(self):
        # a new person must not inherit the previous one's crop, flicker filter or network memory
        self.last_good = None
        self.prev_trans = None
        self.jump_frames = 0
        self.prev_out = None
        for f in self.roi_filters.values():
            f.reset()
        self.box_filter.reset()
        if self.fit is not None:
            self.fit.forget_person()
        active = self.active
        if active is not None and active.kind.recurrent and active.model is not None:
            active.model.reset()

    def _person_changed(self, trans2roi, resolution):
        # Only without tracking. A jump of the crop that persists for JUMP_FRAMES frames means a different person.
        # A single jumpy estimate (raising the arms, leaning) is just smoothed as usual.
        if self.prev_trans is None:
            self.prev_trans = trans2roi
            return False
        prev = self.prev_trans
        scale, prev_scale = np.linalg.norm(trans2roi[0, :2]), np.linalg.norm(prev[0, :2])
        centre = cv2.invertAffineTransform(trans2roi) @ np.array([resolution / 2.0, resolution / 2.0, 1.0])
        prev_centre = cv2.invertAffineTransform(prev) @ np.array([resolution / 2.0, resolution / 2.0, 1.0])
        side = resolution / scale  # crop side in camera pixels
        jumped = (np.linalg.norm(centre - prev_centre) > self.JUMP * side or
                  abs(scale / prev_scale - 1.0) > self.JUMP)
        if not jumped:
            self.jump_frames = 0
            self.prev_trans = trans2roi
            return False
        self.jump_frames += 1
        if self.jump_frames < self.JUMP_FRAMES:
            return False
        self.jump_frames = 0
        self.prev_trans = trans2roi
        return True

    def _roi_filter(self, kind):
        if kind.name not in self.roi_filters:
            self.roi_filters[kind.name] = RoiFilter(kind) if self.smooth else TemporalSmoothing(c=kind.roi_ema)
        return self.roi_filters[kind.name]

    def _box_filter(self, t):
        if not self.smooth:
            return None
        self.box_filter.set_time(t)
        return self.box_filter

    def _check_untouched(self, raw_image, stage):
        # debug only: the camera frame must reach the compositor unchanged; report the stage that draws into it
        if self._pristine is not None and not np.array_equal(self._pristine, raw_image):
            changed = np.any(self._pristine != raw_image, axis=2)
            print('WARNING: the camera frame was modified in place during stage "%s" (%d pixels changed)'
                  % (stage, int(changed.sum())))
            self._pristine = raw_image.copy()

    # ------------------------------------------------------------------ one frame

    def __call__(self, input_frame, ft=NULL_TIMING, t=None):
        # ft: per-frame timing record (util/latency.py); stages are no-ops unless latency logging is on
        # t: time of this frame in seconds (defaults to the clock; pass frame_index / fps for a video file)
        if t is None:
            t = time.perf_counter()
        self.last_debug = None
        self._pristine = input_frame.copy() if self.debug else None
        active = self.active
        if active is None:
            ft.note(outcome='no_garment')
            self.last_good = None
            return input_frame
        kind = active.kind
        roi_h, roi_w = kind.roi_h, kind.roi_w
        raw_image = input_frame
        height, width = raw_image.shape[:2]

        with ft.stage('bev'):
            self.smpl_regressor.set_fix_body(self.BODY_SHAPES[self.body_shape or kind.body_shape])
            # DensePose's image features do not depend on BEV's result, so they are computed meanwhile in a second
            # thread. BEV spends much of its time in Python between small GPU operations, which leaves the GPU free.
            features = (self.pool.submit(self.densepose_extractor.backbone_features, raw_image, self.fp16)
                        if self.pool is not None else None)
            try:
                smpl_param = self.smpl_regressor.forward(raw_image, t=t)
            finally:
                if features is not None:
                    features = features.result()
        self._check_untouched(raw_image, 'bev')
        ft.note(n_people_bev=self.smpl_regressor.n_people_seen)
        if smpl_param is None:
            return self._miss(input_frame, ft, 'no_person')
        trans2roi = kind.crop(self.smpl_regressor, smpl_param)
        if self.smpl_regressor.new_person or (not self.track and self._person_changed(trans2roi, roi_h)):
            self._forget_person()   # snap the crop to the new person instead of sliding across
        roi_filter = self._roi_filter(kind)
        trans2roi, inv_trans2roi = roi_filter(trans2roi, t, height) if self.smooth else roi_filter(trans2roi)
        vertices = torch.from_numpy(SMPL_Regressor.get_raw_verts(smpl_param))

        with ft.stage('vm_render'):
            raw_vm = kind.template().render(vertices, height=height, width=width)

        self._check_untouched(raw_image, 'vm_render')
        with ft.stage('densepose'):
            # DensePose must describe the same person as BEV. The result stays on the GPU (as_tensor): all per-pixel
            # work below runs there.
            joints = SMPL_Regressor.chosen_joints(smpl_param)
            if self.dp_mode == 'box':
                iuv = self.densepose_extractor.get_IUV_in_box(
                    raw_image, SMPL_Regressor.body_box(smpl_param, height, width), target_joints=joints,
                    as_tensor=True, fp16=self.fp16, box_filter=self._box_filter(t), features=features)
            else:
                iuv = self.densepose_extractor.get_IUV(raw_image, isRGB=False, as_tensor=True, target_joints=joints)
        self._check_untouched(raw_image, 'densepose')
        selection = self.densepose_extractor.last_selection
        ft.note(n_people_dp=selection['n_people'], dp_joint_fraction=selection['joint_fraction'])
        if iuv is None:
            return self._miss(input_frame, ft, 'dp_mismatch' if selection['n_people'] else 'dp_no_person')
        iuv = iuv.to(self.device)
        with ft.stage('inputs'):
            vm = torch.from_numpy(raw_vm).to(self.device).permute(2, 0, 1)[None].float()
            # round like the uint8 images the networks were trained on
            roi_vm = gpu_ops.warp_affine(vm, trans2roi, roi_h, roi_w, kind.warp_mode).round()
            roi_map = gpu_ops.warp_affine(kind.body_map(iuv), trans2roi, roi_h, roi_w, kind.warp_mode).round()
            vm_tensor = (roi_vm / 255.0 * 2.0 - 1.0)[:, [2, 1, 0]]  # the VM render is BGR; the network wants RGB
            dp_tensor = roi_map / 255.0 * 2.0 - 1.0
            net_in = torch.cat([vm_tensor, dp_tensor], 1)
        with self.lock:
            if self.active is not active:
                # the garment was switched while this frame was being prepared; the next frame uses the new one
                ft.note(outcome='no_garment')
                return input_frame
            compositor = active.compositor
            with ft.stage('generator'), torch.no_grad():
                p = next(active.model.parameters())
                net_out = active.model.forward(net_in.to(p.dtype)).float()
        with ft.stage('warp_back'):
            if kind.temporal and self.temporal > 0:
                if self.prev_out_t is not None and t - self.prev_out_t > 0.5:
                    self.prev_out = None  # too old to compare with
                net_out = gpu_ops.temporal_filter(net_out, self.prev_out, floor=1.0 - 0.75 * min(self.temporal, 1.0))
                self.prev_out, self.prev_out_t = net_out, t
            net_alpha = kind.alpha(net_out)
            # keep the garment near the body: a network outside its training poses can spread alpha over the
            # whole crop, which shows as a dark, blocky ghost on the background
            roi_alpha = net_alpha
            if self.alpha_gate > 0 and kind.alpha_gate:
                roi_alpha = (net_alpha * gpu_ops.garment_support(roi_vm, roi_map, self.alpha_gate)
                             * gpu_ops.garment_core(net_alpha, roi_vm))
            # BGR colour in 0..255 and alpha in 0..1, warped from the crop back onto the camera frame
            roi_out = torch.cat([((net_out[:, [2, 1, 0]] + 1.0) / 2.0 * 255.0).clamp(0, 255), roi_alpha], 1)
            raw_out = gpu_ops.warp_affine(roi_out, inv_trans2roi, height, width)[0]
            garment, alpha = raw_out[:3].permute(1, 2, 0), raw_out[3]
        self._check_untouched(raw_image, 'inputs/generator/warp_back')
        with ft.stage('composite'):
            occlusion = gpu_ops.hand_head_mask(iuv, parts=self.keep_in_front)
            self.misses = 0
            self.last_good = (garment, alpha, occlusion, compositor)
            composed_img = self._compose(raw_image, garment, alpha, occlusion, compositor, iuv)
        ft.note(outcome='ok')
        if self.debug:
            np_img = lambda x: x.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            raw_IUV = iuv.cpu().numpy()
            occlusion_mask = (occlusion.cpu().numpy() * 255).astype(np.uint8)
            to_roi = lambda img, flags=cv2.INTER_LINEAR: cv2.warpAffine(img, trans2roi, (roi_w, roi_h), flags=flags)
            self.last_debug = {'input': to_roi(raw_image), 'vm': np_img(roi_vm[0].permute(1, 2, 0)),
                               'sdp': np_img(roi_map[0].permute(1, 2, 0)),
                               'garment': np_img(roi_out[0, :3].permute(1, 2, 0)),
                               'alpha': roi_out[0, 3].cpu().numpy(),
                               'alpha_net': net_alpha[0, 0].cpu().numpy(),  # before the body gate
                               'occlusion': to_roi(occlusion_mask, cv2.INTER_NEAREST),
                               # full-frame arrays, saved with S for offline analysis
                               'raw': {'frame': raw_image, 'garment': np_img(garment), 'alpha': alpha.cpu().numpy(),
                                       'occlusion': occlusion_mask, 'iuv': raw_IUV, 'output': composed_img}}
        return composed_img
