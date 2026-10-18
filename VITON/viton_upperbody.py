import os
import threading

import numpy as np
import cv2

import util.util
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
import util.util as util
import torch
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from SMPL.smpl_regressor import SMPL_Regressor
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from util.image_process import blur_image
#from composition.short_sleeve_composition import ShortSleeveComposer
from model.DensePose.densepose_extractor import DensePoseExtractor
import time

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP
from threading import Thread
import torch.nn.functional as F
from util.cv2_trans_util import TemporalSmoothing
from composition.garment_removal import WornGarmentRemover


def make_pix2pix_model(name, input_nc, output_nc=3, model_name='pix2pixHD',ckpt_dir=None):
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = name
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.isTrain = False
    opt.model = model_name
    if ckpt_dir is None:
        opt.checkpoints_dir='./rtv_ckpts'
    else:
        opt.checkpoints_dir=ckpt_dir
    opt.gpu_ids=''#load to cpu
    model = create_model(opt)
    # print(model)
    return model


class FrameProcessor:
    def __init__(self, garment_name_list,ckpt_dir=None):
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.ckpt_dir = ckpt_dir
        # RTV_DP_SIZE: short side DensePose runs at (detectron2 default is 800). Smaller is faster.
        dp_size = int(os.environ.get('RTV_DP_SIZE', '448'))
        self.densepose_extractor = DensePoseExtractor(opts=[
            'INPUT.MIN_SIZE_TEST', dp_size, 'INPUT.MAX_SIZE_TEST', dp_size * 2,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 300, 'MODEL.RPN.POST_NMS_TOPK_TEST', 50])
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list#[2, 3, 17, 18, 22]
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        self.lock = threading.Lock()
        self.roi_smoothing = TemporalSmoothing(c=0.5)
        # RTV_REMOVE_CLOTHES=0 goes back to overlaying the garment on top of the worn one
        self.remove_worn = os.environ.get('RTV_REMOVE_CLOTHES', '1') != '0'
        self.garment_remover = WornGarmentRemover()
        self.grid_cache = {}
        self.sdp_lut = None

        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def load_all_models(self):
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = make_pix2pix_model(garment_name, 6, output_nc=4,ckpt_dir=self.ckpt_dir)
            if self.viton_model_list[i] is None:
                self.viton_model_list[i]= new_model

    def load_one_models(self, garment_name):
        new_model = make_pix2pix_model(garment_name, 6, output_nc=4,ckpt_dir=self.ckpt_dir)
        id = self.garment_name_list.index(garment_name)
        if self.viton_model_list[id] is None:
            self.viton_model_list[id]= new_model

    def switch_to_target_garment(self,garment_id):
        self.lock.acquire()
        print("Loading from CPU target garment id: ", garment_id)
        id = garment_id
        if self.viton_model_list[id] is None and id >= 0:
            print("Loading from disk target garment id: ", garment_id)
            self.load_one_models(self.garment_name_list[id])
        old_model = self.viton_model
        new_model=self.viton_model_list[id].to('cuda:0') if garment_id>=0 else None
        print("Finished")
        if self.viton_model is not None:
            del self.viton_model
        self.viton_model = new_model
        if old_model is not None:
            old_model = old_model.to('cpu')
            del old_model
            torch.cuda.empty_cache()
        self.lock.release()


    def set_target_garment(self, target_id):
        #new_model = make_pix2pix_model(ckpt_dict[target_id], 6, output_nc=4)
        #self.viton_model = self.viton_model_list[target_id]
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
    def __call__(self, input_frame):
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame
        height, width = raw_image.shape[:2]

        smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                roi_img_size=resolution)
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

        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        raw_vm = self.upper_body.render(torch.from_numpy(vertices), height=height, width=width)
        vm = torch.from_numpy(raw_vm).cuda().permute(2, 0, 1)[None].float() / 255.0
        sdp, labels = self._iuv2sdp(raw_IUV)

        vm_tensor = self._sample(vm, inv_trans2roi, resolution, resolution) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
        dp_tensor = self._sample(sdp, inv_trans2roi, resolution, resolution) * 2.0 - 1.0
        with self.lock:
            if self.viton_model is None:
                return input_frame
            with torch.autocast('cuda', dtype=torch.float16):
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1))
        target_tensor = target_tensor.float()
        # BGR colour in [0, 255] plus coverage in [0, 1], warped back onto the full frame
        roi_target = torch.cat([(target_tensor[:, [2, 1, 0]] + 1.0) * 127.5, target_tensor[:, 3:4]], 1)
        raw_target = self._sample(roi_target.clamp(0.0, 255.0), trans2roi, height, width)[0]
        garment = raw_target[:3].permute(1, 2, 0)
        alpha = soft_alpha(raw_target[3])

        frame = torch.from_numpy(raw_image).cuda().float()
        if self.remove_worn:
            roi_side = resolution / float(np.linalg.norm(trans2roi[0, :2]))
            frame = self.garment_remover(frame, labels, vm[0].amax(0) > 0, alpha, roi_side)
        composed = torch.lerp(frame, garment, alpha[..., None])
        return composed.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()


def soft_alpha(alpha):
    # alpha: (H, W) raw coverage. Shrink by a pixel to drop the dark fringe, then keep a soft edge
    # instead of the hard >0.5 cut-off, which is what made garment borders look jagged.
    alpha = -F.max_pool2d(-alpha[None, None], 3, 1, 1)[0, 0]
    return ((alpha - 0.3) / 0.4).clamp(0.0, 1.0)
