import os
import threading

import numpy as np
import cv2

import util.util as util
import torch
from SMPL.smpl_regressor import SMPL_Regressor
from model.DensePose.densepose_extractor import DensePoseExtractor
from model.garment_model import make_pix2pix_model, read_meta

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
import torch.nn.functional as F
from composition.compositor import GarmentCompositor
from composition.garment_removal import WornGarmentRemover
from util.cv2_trans_util import TemporalSmoothing
from threading import Thread
from util.latency import NULL_TIMING
from util.gpu_check import report_devices


class FrameProcessor:
    def __init__(self, garment_name_list, ckpt_dir=None):
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.compositor = None
        self.ckpt_dir = ckpt_dir if ckpt_dir is not None else './rtv_ckpts'
        # RTV_DP_SIZE: short side DensePose runs at (detectron2 default is 800). Smaller is faster.
        dp_size = int(os.environ.get('RTV_DP_SIZE', '448'))
        self.densepose_extractor = DensePoseExtractor(opts=[
            'INPUT.MIN_SIZE_TEST', dp_size, 'INPUT.MAX_SIZE_TEST', dp_size * 2,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 300, 'MODEL.RPN.POST_NMS_TOPK_TEST', 50])
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        # how each checkpoint's colour was trained decides how it is composited
        self.compositor_list = [GarmentCompositor(premultiplied=read_meta(self.ckpt_dir, name)['color'] == 'premultiplied')
                                for name in self.garment_name_list]
        # the ROI only frames the network input (output is warped back with its inverse), so smoothing it
        # removes jitter from noisy joints without misaligning the garment
        self.roi_smoothing = TemporalSmoothing(c=0.5)
        self.lock = threading.Lock()
        # RTV_REMOVE_CLOTHES=0 goes back to overlaying the garment on top of the worn one
        self.remove_worn = os.environ.get('RTV_REMOVE_CLOTHES', '1') != '0'
        self.garment_remover = WornGarmentRemover()
        self.grid_cache = {}
        self.sdp_lut = None
        self.debug = False       # when True, keep this frame's network inputs/outputs in self.last_debug
        self.last_debug = None
        print('Model devices:')
        report_devices({'BEV': self.smpl_regressor.regressor_model,
                        'DensePose': self.densepose_extractor.predictor.model})

        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def make_model(self, garment_name):
        return make_pix2pix_model(garment_name, 6, output_nc=4, model_name='pix2pixHD', ckpt_dir=self.ckpt_dir)

    def load_all_models(self):
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = self.make_model(garment_name)
            if self.viton_model_list[i] is None:
                self.viton_model_list[i] = new_model

    def load_one_models(self, garment_name):
        new_model = self.make_model(garment_name)
        id = self.garment_name_list.index(garment_name)
        if self.viton_model_list[id] is None:
            self.viton_model_list[id] = new_model

    def switch_to_target_garment(self, garment_id):
        with self.lock:
            print("Loading from CPU target garment id: ", garment_id)
            id = garment_id
            if id >= 0 and self.viton_model_list[id] is None:
                print("Loading from disk target garment id: ", garment_id)
                self.load_one_models(self.garment_name_list[id])
            old_model = self.viton_model
            new_model = self.viton_model_list[id].to('cuda:0') if garment_id >= 0 else None
            print("Finished; garment network on", next(new_model.parameters()).device if new_model is not None else None)
            self.viton_model = new_model
            self.compositor = self.compositor_list[id] if garment_id >= 0 else None
            if old_model is not None and old_model is not new_model:
                old_model.to('cpu')
                torch.cuda.empty_cache()

    def set_target_garment(self, target_id):
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    def _sample(self, src, dst2src, out_h, out_w, mode='bilinear'):
        # GPU equivalent of cv2.warpAffine: src (1, C, h, w), dst2src maps output pixels to source pixels
        key = (out_h, out_w)
        if key not in self.grid_cache:
            ys, xs = torch.meshgrid(torch.arange(out_h, device='cuda', dtype=torch.float32),
                                    torch.arange(out_w, device='cuda', dtype=torch.float32), indexing='ij')
            self.grid_cache[key] = (xs, ys)
        xs, ys = self.grid_cache[key]
        m = [[float(v) for v in row] for row in dst2src]
        h, w = src.shape[-2:]
        sx = m[0][0] * xs + m[0][1] * ys + m[0][2]
        sy = m[1][0] * xs + m[1][1] * ys + m[1][2]
        grid = torch.stack([2.0 * sx / (w - 1) - 1.0, 2.0 * sy / (h - 1) - 1.0], -1)[None]
        return F.grid_sample(src, grid, mode=mode, padding_mode='zeros', align_corners=True)

    def _iuv2sdp(self, iuv):
        # GPU version of util.densepose_util.IUV2SDP; iuv: uint8 (H, W, 3) -> float (1, 3, H, W) in [0, 1]
        if self.sdp_lut is None:
            self.sdp_lut = torch.zeros(25, dtype=torch.bool, device=iuv.device)
            self.sdp_lut[[1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14]] = True
        labels = iuv[:, :, 0].long()
        sdp = iuv.float()
        sdp[:, :, 0] = torch.floor(sdp[:, :, 0] / 24 * 255)
        sdp[self.sdp_lut[labels]] = 255
        return sdp.permute(2, 0, 1)[None] / 255.0, labels

    def reset_background(self):
        self.garment_remover.reset_background()

    @torch.no_grad()
    def __call__(self, input_frame, ft=NULL_TIMING):
        # ft: per-frame timing record (util/latency.py); stages are no-ops unless latency logging is on
        self.last_debug = None
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame
        height, width = raw_image.shape[:2]

        with ft.stage('bev'):
            smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                    roi_img_size=resolution)
        with ft.stage('densepose'):
            raw_IUV = self.densepose_extractor.get_IUV_tensor(raw_image)
        smpl_param = smpl_data[0] if len(smpl_data) >= 3 else None
        if smpl_param is None:
            self.roi_smoothing.past_trans = None
            if raw_IUV is None and self.remove_worn:
                # nobody in view: a clean look at the background
                self.garment_remover.observe_empty(torch.from_numpy(raw_image).cuda().float())
            return input_frame
        if raw_IUV is None:
            return input_frame
        trans2roi, inv_trans2roi = self.roi_smoothing(smpl_data[1])

        with ft.stage('vm_render'):
            vertices = SMPL_Regressor.get_raw_verts(smpl_param)
            raw_vm = self.upper_body.render(torch.from_numpy(vertices), height=height, width=width)
        with ft.stage('inputs'):
            vm = torch.from_numpy(raw_vm).cuda().permute(2, 0, 1)[None].float() / 255.0
            sdp, labels = self._iuv2sdp(raw_IUV)
            vm_tensor = self._sample(vm, inv_trans2roi, resolution, resolution) * 2.0 - 1.0
            vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
            dp_tensor = self._sample(sdp, inv_trans2roi, resolution, resolution) * 2.0 - 1.0
        with self.lock:
            if self.viton_model is None:
                return input_frame
            compositor = self.compositor
            with ft.stage('generator'), torch.autocast('cuda', dtype=torch.float16):
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1))
        with ft.stage('warp_back'):
            target_tensor = target_tensor.float()
            # BGR colour in [0, 255] plus coverage in [0, 1], warped back onto the full frame
            roi_target = torch.cat([(target_tensor[:, [2, 1, 0]] + 1.0) * 127.5, target_tensor[:, 3:4]], 1)
            raw_target = self._sample(roi_target.clamp(0.0, 255.0), trans2roi, height, width)[0]
        with ft.stage('composite'):
            garment, alpha = composite_inputs(compositor, raw_target[:3].permute(1, 2, 0), raw_target[3])
            # keep hands and face in front of the garment
            occlusion = hand_head_mask(labels)
            alpha = alpha * (1.0 - occlusion)
            frame = torch.from_numpy(raw_image).cuda().float()
            if self.remove_worn:
                roi_side = resolution / float(np.linalg.norm(trans2roi[0, :2]))
                frame = self.garment_remover(frame, labels, vm[0].amax(0) > 0, alpha, roi_side)
            composed = torch.lerp(frame, garment, alpha[..., None])
            composed = composed.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        if self.debug:
            to_img = lambda t: t[0].permute(1, 2, 0).mul(255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
            roi_occlusion = self._sample(occlusion[None, None], inv_trans2roi, resolution, resolution)
            self.last_debug = {'vm': to_img(vm_tensor[:, [2, 1, 0]] * 0.5 + 0.5), 'sdp': to_img(dp_tensor * 0.5 + 0.5),
                               'alpha': target_tensor[0, 3].clamp(0.0, 1.0).cpu().numpy(),
                               'occlusion': to_img(roi_occlusion)[..., 0]}
        return composed


def composite_inputs(compositor, garment, alpha, eps=1e-3):
    # GPU version of composition.compositor.GarmentCompositor's colour/alpha handling.
    # garment: (H, W, 3) in [0, 255]; alpha: (H, W) raw coverage -> straight colour and soft alpha
    alpha = alpha.clamp(0.0, 1.0)
    if compositor.premultiplied:
        # the network learned colour darkened by alpha along the edge; undo it to drop the dark outline
        garment = (garment / alpha.clamp(min=eps)[..., None]).clamp(0.0, 255.0) * (alpha > eps)[..., None]
    a = ((alpha - compositor.alpha_low) / (compositor.alpha_high - compositor.alpha_low)).clamp(0.0, 1.0)
    return garment, a


def hand_head_mask(labels):
    # GPU version of util.densepose_util.IUV2HandHeadMask (+ the compositor's feathering): DensePose
    # hands (3, 4) and head (23, 24), grown 2px, then lightly blurred so the soft edge sits on their boundary.
    # labels: (H, W) long -> (H, W) float in [0, 1]
    mask = ((labels == 3) | (labels == 4) | (labels == 23) | (labels == 24)).float()[None, None]
    mask = F.max_pool2d(mask, 5, 1, 2)
    return F.avg_pool2d(mask, 5, 1, 2)[0, 0]
