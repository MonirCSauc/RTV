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

from SMPL.fullbody_smpl.FullBody import FullBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP, IUV2SSDP, IUV2HandHeadMask
from threading import Thread
from util.cv2_trans_util import TemporalSmoothing
from model.garment_model import make_pix2pix_model, read_meta
from composition.compositor import GarmentCompositor

class FullBodySeqFrameProcessor:
    def __init__(self, target_name='coat_seq_vmssdp2ta_576'):
        self.viton_model = make_pix2pix_model(target_name, model_name='pix2pixHD_RNN_RGBA', on_gpu=True)
        self.compositor = GarmentCompositor(premultiplied=read_meta('./rtv_ckpts', target_name)['color'] == 'premultiplied')
        self.smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
        self.full_body = FullBodySMPL()
        self.temporal_smoothing = TemporalSmoothing(c=0.9)
        self.roi_height = 576
        self.roi_width = int(self.roi_height * 0.75)
        self.densepose_extractor = DensePoseExtractor(opts=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5])
        self.misses = 0

    # after this many frames without the person, the next one is a new customer: the ConvLSTM memory and the crop
    # smoothing must not carry over from the previous person
    LOST_AFTER = 15

    def _miss(self, raw_image):
        self.misses += 1
        if self.misses == self.LOST_AFTER:
            self.temporal_smoothing.reset()
            self.viton_model.reset()
        return raw_image

    def __call__(self, roi_vm, roi_ssdp):
        vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
        dp_tensor = util.im2tensor(roi_ssdp) * 2.0 - 1.0
        with torch.no_grad():
            target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1).cuda())
        roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
        roi_alpha = ((target_tensor[0, 3, :, :].clamp(min=-1.0, max=1.0) / 2.0 + 0.5).cpu().numpy() * 255).astype(
            np.uint8)
        return roi_target, roi_alpha

    def forward(self, raw_image, isRGB=False):
        height = raw_image.shape[0]
        width = raw_image.shape[1]

        smpl_param = self.smpl_regressor.forward(raw_image, False)  # 1.38
        # print(list(smpl_param.keys()))
        if smpl_param is None:
            return self._miss(raw_image)
        trans2roi, inv_trans = self.smpl_regressor.get_fullbody_trans2roi(smpl_param, s=1.4, new_h=self.roi_height,
                                                                          new_w=self.roi_width)
        trans2roi, inv_trans = self.temporal_smoothing(trans2roi)

        vertices = self.smpl_regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        v = vertices

        raw_vm = self.full_body.render(v[0], height=height, width=width)
        roi_vm = cv2.warpAffine(raw_vm, trans2roi, (self.roi_width, self.roi_height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        # DensePose must describe the same person as BEV; an all-black body map (the old fallback) is not a person
        IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False,
                                               target_joints=SMPL_Regressor.chosen_joints(smpl_param))
        if IUV is None:
            return self._miss(raw_image)
        self.misses = 0

        ssdp = IUV2SSDP(IUV)
        roi_ssdp = cv2.warpAffine(ssdp, trans2roi, (self.roi_width, self.roi_height), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

        roi_target, roi_alpha = self.__call__(roi_vm, roi_ssdp)
        raw_target_img = cv2.warpAffine(roi_target, inv_trans, (raw_image.shape[1], raw_image.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        raw_alpha = cv2.warpAffine(roi_alpha, inv_trans, (raw_image.shape[1], raw_image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))
        occlusion_mask = IUV2HandHeadMask(IUV)
        composed_img = self.compositor(raw_image, raw_target_img, raw_alpha, occlusion_mask)

        return composed_img
