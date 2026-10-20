"""Compare per-stage inference time of demo.py (full-body seq) and rtl_demo.py (upper-body) on the same frames.

Usage:
    python profile_demos.py --camera 0
    python profile_demos.py --camera http://<phone-ip>:4747/video --rotate none
"""
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from util.image_warp import crop2_169, resize_img


def capture_frames(source, rotate, num_frames):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")
    frames = []
    start = time.perf_counter()
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        if rotate == 'cw':
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 'ccw':
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        frame = resize_img(frame, max_height=1024)
        frame = crop2_169(frame)
        frames.append(frame)
    elapsed = time.perf_counter() - start
    cap.release()
    print(f"Camera read: {num_frames / elapsed:.1f} FPS (raw capture, no processing), frame size {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


def timed(stats, name, fn):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        stats[name].append(time.perf_counter() - t)
        return result
    return wrapper


def run(name, process, frames, stats, warmup):
    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        t = time.perf_counter()
        process(frame)
        torch.cuda.synchronize()
        if i >= warmup:
            stats['TOTAL'].append(time.perf_counter() - t)
        else:
            for k in list(stats.keys()):
                if k != 'TOTAL':
                    stats[k].clear()

    print(f"\n=== {name} ===")
    for k, v in stats.items():
        if v:
            print(f"  {k:<24} {np.mean(v) * 1000:7.1f} ms  (n={len(v)})")
    if stats['TOTAL']:
        print(f"  -> {1.0 / np.mean(stats['TOTAL']):.1f} FPS")


def profile_fullbody_seq(frames, warmup):
    import VITON.viton_fullbody_seq as m
    stats = defaultdict(list)
    proc = m.FullBodySeqFrameProcessor('coat_seq_vmssdp2ta_576')
    proc.smpl_regressor.forward = timed(stats, 'BEV (SMPL regressor)', proc.smpl_regressor.forward)
    proc.full_body.render = timed(stats, 'render body mesh', proc.full_body.render)
    proc.densepose_extractor.get_IUV = timed(stats, 'DensePose', proc.densepose_extractor.get_IUV)
    m.IUV2SSDP = timed(stats, 'IUV2SSDP', m.IUV2SSDP)
    proc.viton_model.forward = timed(stats, 'try-on network', proc.viton_model.forward)
    run('demo.py  (full-body, pix2pixHD_RNN_RGBA)', proc.forward, frames, stats, warmup)
    del proc
    torch.cuda.empty_cache()


def profile_upperbody(frames, warmup, garment):
    import VITON.viton_upperbody as m
    stats = defaultdict(list)
    proc = m.FrameProcessor([garment])
    proc.switch_to_target_garment(0)
    proc.smpl_regressor.forward = timed(stats, 'BEV (SMPL regressor)', proc.smpl_regressor.forward)
    proc.upper_body.render = timed(stats, 'render body mesh', proc.upper_body.render)
    proc.densepose_extractor.get_IUV = timed(stats, 'DensePose', proc.densepose_extractor.get_IUV)
    m.IUV2SDP = timed(stats, 'IUV2SDP', m.IUV2SDP)
    proc.viton_model.forward = timed(stats, 'try-on network', proc.viton_model.forward)
    run(f'rtl_demo.py  (upper-body, pix2pixHD, {garment})', proc, frames, stats, warmup)
    del proc
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='0')
    parser.add_argument('--rotate', choices=['cw', 'ccw', 'none'], default='cw')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--garment', type=str, default='lab_03_vmsdp2ta')
    args = parser.parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera

    print("Stand fully in view of the camera while frames are captured...")
    frames = capture_frames(source, args.rotate, args.frames)
    profile_upperbody(frames, args.warmup, args.garment)
    profile_fullbody_seq(frames, args.warmup)
