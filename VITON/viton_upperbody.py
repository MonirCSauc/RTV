import threading

import numpy as np
import cv2

import util.util as util
import torch
from SMPL.smpl_regressor import SMPL_Regressor
from model.DensePose.densepose_extractor import DensePoseExtractor
from model.garment_model import make_pix2pix_model, read_meta

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from composition.compositor import GarmentCompositor
from util.densepose_util import IUV2SDP, IUV2HandHeadMask
from util.cv2_trans_util import TemporalSmoothing
from threading import Thread
from util.latency import NULL_TIMING
from util.gpu_check import report_devices


class FrameProcessor:
    # --body_shape: 'train' zeroes the shape parameters exactly as the training data did (default, matches the
    # released checkpoints); 'zero' zeroes all of them (paper); 'estimated' keeps BEV's per-person estimate, which
    # a coat inflates
    BODY_SHAPES = {'train': True, 'zero': 'all', 'estimated': False}

    def __init__(self, garment_name_list, ckpt_dir=None, dp_score=0.5, body_shape='train'):
        self.smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=self.BODY_SHAPES[body_shape])
        self.viton_model = None
        self.compositor = None
        self.ckpt_dir = ckpt_dir if ckpt_dir is not None else './rtv_ckpts'
        # detectron2 keeps detections down to a 0.05 score by default, so posters, mannequins and reflections compete
        # for "the person"; dp_score drops weak detections
        self.densepose_extractor = DensePoseExtractor(opts=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', dp_score])
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        # how each checkpoint's colour was trained decides how it is composited
        self.compositor_list = [GarmentCompositor(premultiplied=read_meta(self.ckpt_dir, name)['color'] == 'premultiplied')
                                for name in self.garment_name_list]
        # the ROI only frames the network input (output is warped back with its inverse), so smoothing it
        # removes jitter from noisy joints without misaligning the garment
        self.roi_smoothing = TemporalSmoothing(c=0.6)
        # Misses (no person, or DensePose disagrees with BEV): show the last garment for HOLD frames, fade it out over
        # FADE frames, then treat the person as gone. A newly found person or garment fades in over FADE_IN frames.
        self.misses = 0
        self.last_good = None      # (garment, alpha, occlusion, compositor) of the last successful frame
        self.visibility = 0.0      # overall garment opacity, 0..1
        self.prev_trans = None     # unsmoothed crop transform of the previous frame, to detect a different person
        self.lock = threading.Lock()
        self.debug = False       # when True, keep this frame's network inputs/outputs in self.last_debug
        self.last_debug = None
        self._pristine = None
        print('Model devices:')
        report_devices({'BEV': self.smpl_regressor.regressor_model,
                        'DensePose': self.densepose_extractor.predictor.model})

        self.resident = False
        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def make_model(self, garment_name):
        return make_pix2pix_model(garment_name, 6, output_nc=4, model_name='pix2pixHD', ckpt_dir=self.ckpt_dir)

    GARMENT_GPU_BYTES = 800 * 2 ** 20  # one fp32 pix2pixHD garment network is ~730 MB

    def fits_on_gpu(self, n_models):
        # keep every garment resident on the GPU when it leaves room for BEV, DensePose and activations (~2.5 GB),
        # so switching garments is a pointer swap instead of a 730 MB copy while the frame loop waits
        free, _ = torch.cuda.mem_get_info()
        return free > n_models * self.GARMENT_GPU_BYTES + 2.5 * 2 ** 30

    def load_all_models(self):
        resident = torch.cuda.is_available() and self.fits_on_gpu(len(self.garment_name_list))
        print('Garment networks: %s' % ('all kept on the GPU (instant switching)' if resident else
                                        'kept in CPU memory, copied to the GPU when selected'))
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = self.make_model(garment_name)
            if resident:
                new_model = new_model.to('cuda:0')
            if self.viton_model_list[i] is None:
                self.viton_model_list[i] = new_model
        self.resident = resident

    def load_one_models(self, garment_name):
        new_model = self.make_model(garment_name)
        id = self.garment_name_list.index(garment_name)
        if self.viton_model_list[id] is None:
            self.viton_model_list[id] = new_model

    def switch_to_target_garment(self, garment_id):
        # load and copy to the GPU outside the lock: the frame loop keeps running on the current garment meanwhile
        new_model = None
        if garment_id >= 0:
            if self.viton_model_list[garment_id] is None:
                print("Loading from disk target garment id: ", garment_id)
                self.load_one_models(self.garment_name_list[garment_id])
            new_model = self.viton_model_list[garment_id].to('cuda:0')
        print("Garment %d ready on %s" % (garment_id, next(new_model.parameters()).device if new_model is not None
                                          else None))
        with self.lock:  # only the swap itself blocks the frame loop
            old_model = self.viton_model
            self.viton_model = new_model
            self.compositor = self.compositor_list[garment_id] if garment_id >= 0 else None
            self.last_good = None  # never hold or fade the previous garment's image
            self.visibility = 0.0
        if old_model is not None and old_model is not new_model and not self.resident:
            old_model.to('cpu')
            torch.cuda.empty_cache()

    def set_target_garment(self, target_id):
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    HOLD, FADE, FADE_IN = 3, 12, 4
    JUMP = 0.25  # crop centre moving more than this share of the crop side, or its scale changing this much

    def _miss(self, input_frame, ft, outcome):
        # this frame produced no garment: hold, then fade, the last one instead of blinking
        ft.note(outcome=outcome)
        self.misses += 1
        if self.last_good is not None and self.misses <= self.HOLD + self.FADE:
            fade = 1.0 if self.misses <= self.HOLD else 1.0 - (self.misses - self.HOLD) / float(self.FADE)
            garment, alpha, occlusion, compositor = self.last_good
            return compositor(input_frame, garment, alpha, occlusion, global_alpha=self.visibility * fade)
        if self.last_good is not None or self.misses == self.HOLD + self.FADE + 1:
            self._forget_person()
        return input_frame

    def _forget_person(self):
        self.last_good = None
        self.visibility = 0.0
        self.prev_trans = None
        self.roi_smoothing.reset()

    def _person_changed(self, trans2roi, resolution):
        # a big jump of the crop between consecutive frames means a different person (or a lost track)
        prev, self.prev_trans = self.prev_trans, trans2roi
        if prev is None:
            return False
        scale, prev_scale = np.linalg.norm(trans2roi[0, :2]), np.linalg.norm(prev[0, :2])
        centre = cv2.invertAffineTransform(trans2roi) @ np.array([resolution / 2.0, resolution / 2.0, 1.0])
        prev_centre = cv2.invertAffineTransform(prev) @ np.array([resolution / 2.0, resolution / 2.0, 1.0])
        side = resolution / scale  # crop side in camera pixels
        return (np.linalg.norm(centre - prev_centre) > self.JUMP * side or
                abs(scale / prev_scale - 1.0) > self.JUMP)

    def _check_untouched(self, raw_image, stage):
        # debug only: the camera frame must reach the compositor unchanged; report the stage that draws into it
        if self._pristine is not None and not np.array_equal(self._pristine, raw_image):
            changed = np.any(self._pristine != raw_image, axis=2)
            print('WARNING: the camera frame was modified in place during stage "%s" (%d pixels changed)'
                  % (stage, int(changed.sum())))
            self._pristine = raw_image.copy()

    def __call__(self, input_frame, ft=NULL_TIMING):
        # ft: per-frame timing record (util/latency.py); stages are no-ops unless latency logging is on
        self.last_debug = None
        self._pristine = input_frame.copy() if self.debug else None
        if self.viton_model is None:
            ft.note(outcome='no_garment')
            self._forget_person()
            return input_frame

        resolution = 512
        raw_image = input_frame

        with ft.stage('bev'):
            smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                    roi_img_size=resolution)
        self._check_untouched(raw_image, 'bev')
        if len(smpl_data) < 3 or smpl_data[0] is None:
            ft.note(n_people_bev=0)
            return self._miss(input_frame, ft, 'no_person')
        smpl_param, trans2roi, inv_trans2roi = smpl_data
        ft.note(n_people_bev=SMPL_Regressor.n_people(smpl_param))
        if self._person_changed(trans2roi, resolution):
            self._forget_person()   # snap the crop to the new person and fade the garment in again
            self.prev_trans = trans2roi
        trans2roi, inv_trans2roi = self.roi_smoothing(trans2roi)
        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        height = raw_image.shape[0]
        width = raw_image.shape[1]

        v = vertices

        with ft.stage('vm_render'):
            raw_vm = self.upper_body.render(v[0], height=height, width=width)

        self._check_untouched(raw_image, 'vm_render')
        with ft.stage('densepose'):
            # DensePose must describe the same person as BEV: pick its detection by BEV's 2D joints
            raw_IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False,
                                                       target_joints=SMPL_Regressor.chosen_joints(smpl_param))
        self._check_untouched(raw_image, 'densepose')
        selection = self.densepose_extractor.last_selection
        ft.note(n_people_dp=selection['n_people'], dp_joint_fraction=selection['joint_fraction'])
        if raw_IUV is None:
            return self._miss(input_frame, ft, 'dp_mismatch' if selection['n_people'] else 'dp_no_person')
        with ft.stage('inputs'):
            dpi_img = IUV2SDP(raw_IUV)
            roi_dpi_img = cv2.warpAffine(dpi_img, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
            roi_vm = cv2.warpAffine(raw_vm, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
            vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
            vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
            dp_tensor = util.im2tensor(roi_dpi_img) * 2.0 - 1.0
        with self.lock:
            if self.viton_model is None:
                ft.note(outcome='no_garment')
                return input_frame
            compositor = self.compositor
            with ft.stage('generator'), torch.no_grad():
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1).cuda())
        with ft.stage('warp_back'):
            roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
            roi_alpha = target_tensor[0, 3, :, :].clamp(min=0.0, max=1.0).cpu().numpy().astype(np.float32)
            raw_target_img = cv2.warpAffine(roi_target, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
            raw_alpha = cv2.warpAffine(roi_alpha, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0,))
        self._check_untouched(raw_image, 'inputs/generator/warp_back')
        with ft.stage('composite'):
            occlusion_mask = IUV2HandHeadMask(raw_IUV)
            self.misses = 0
            self.visibility = min(1.0, self.visibility + 1.0 / self.FADE_IN)
            self.last_good = (raw_target_img, raw_alpha, occlusion_mask, compositor)
            composed_img = compositor(raw_image, raw_target_img, raw_alpha, occlusion_mask,
                                      global_alpha=self.visibility)
        ft.note(outcome='ok')
        if self.debug:
            to_roi = lambda img, flags=cv2.INTER_LINEAR: cv2.warpAffine(img, trans2roi, (resolution, resolution), flags=flags)
            self.last_debug = {'input': to_roi(raw_image), 'vm': roi_vm, 'sdp': roi_dpi_img,
                               'garment': roi_target, 'alpha': roi_alpha,
                               'occlusion': to_roi(occlusion_mask, cv2.INTER_NEAREST),
                               # full-frame arrays, saved with S for offline analysis
                               'raw': {'frame': raw_image, 'garment': raw_target_img, 'alpha': raw_alpha,
                                       'occlusion': occlusion_mask, 'iuv': raw_IUV, 'output': composed_img}}
        return composed_img
