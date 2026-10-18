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


class FrameProcessor:
    def __init__(self, garment_name_list, ckpt_dir=None):
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.compositor = None
        self.ckpt_dir = ckpt_dir if ckpt_dir is not None else './rtv_ckpts'
        self.densepose_extractor = DensePoseExtractor()
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        # how each checkpoint's colour was trained decides how it is composited
        self.compositor_list = [GarmentCompositor(premultiplied=read_meta(self.ckpt_dir, name)['color'] == 'premultiplied')
                                for name in self.garment_name_list]
        # the ROI only frames the network input (output is warped back with its inverse), so smoothing it
        # removes jitter from noisy joints without misaligning the garment
        self.roi_smoothing = TemporalSmoothing(c=0.6)
        self.lock = threading.Lock()

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
            print("Finished")
            self.viton_model = new_model
            self.compositor = self.compositor_list[id] if garment_id >= 0 else None
            if old_model is not None and old_model is not new_model:
                old_model.to('cpu')
                torch.cuda.empty_cache()

    def set_target_garment(self, target_id):
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    def __call__(self, input_frame):
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame

        smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                roi_img_size=resolution)
        if len(smpl_data) < 3:
            return input_frame
        smpl_param, trans2roi, inv_trans2roi = smpl_data

        if smpl_param is None:
            return input_frame
        trans2roi, inv_trans2roi = self.roi_smoothing(trans2roi)
        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        height = raw_image.shape[0]
        width = raw_image.shape[1]

        v = vertices

        raw_vm = self.upper_body.render(v[0], height=height, width=width)

        raw_IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False)
        if raw_IUV is None:
            return input_frame
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
                return input_frame
            compositor = self.compositor
            with torch.no_grad():
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1).cuda())
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
        occlusion_mask = IUV2HandHeadMask(raw_IUV)
        composed_img = compositor(raw_image, raw_target_img, raw_alpha, occlusion_mask)
        return composed_img
