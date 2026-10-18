import json
import os

from options.test_options import TestOptions
from model.pix2pixHD.models import create_model

META_FILE = 'rtv_meta.json'

# Checkpoints without rtv_meta.json (e.g. the released rtv_ckpts) were trained on garment crops cut out
# on a black background, so their colour output is premultiplied by alpha.
DEFAULT_META = {'color': 'premultiplied'}


def write_meta(ckpt_dir, name, meta):
    path = os.path.join(ckpt_dir, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, META_FILE), 'w') as f:
        json.dump(meta, f, indent=2)


def read_meta(ckpt_dir, name):
    meta = dict(DEFAULT_META)
    path = os.path.join(ckpt_dir, name, META_FILE)
    if os.path.exists(path):
        with open(path) as f:
            meta.update(json.load(f))
    return meta


def make_pix2pix_model(name, input_nc=6, output_nc=4, model_name='pix2pixHD_RGBA', ckpt_dir='./rtv_ckpts',
                       on_gpu=False):
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
    opt.checkpoints_dir = ckpt_dir
    opt.gpu_ids = ''  # load to cpu; callers move the model to the GPU when it is selected
    model = create_model(opt)
    return model.cuda() if on_gpu else model
