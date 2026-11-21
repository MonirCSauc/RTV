"""Run the try-on pipeline on a video file and measure speed and temporal stability.

    python tools/benchmark_video.py --video in.mp4 --garment lab_03_vmsdp2ta [--out out.mp4] [pipeline options]

Reports, on the same frames for every setting (so settings can be compared, e.g. with and without --original):

- Per-stage time and frame rate (GPU work is synchronised per stage, as in --latency_csv).
- Warping error (flicker beyond real motion): the previous output frame is moved onto the current one with the
  optical flow of the *camera* frames, and the remaining difference inside the garment is averaged. The same
  number for the camera frames themselves is the floor that real motion, lighting and flow errors cause; the
  excess over it is flicker added by the try-on. Lower is better. Values are in grey levels (0..255).
- Frame outcomes (how often the garment was drawn).
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import cv2
import numpy as np
import torch

from VITON.frame_processor import FrameProcessor
from VITON.options import add_pipeline_arguments, processor_options
from util.latency import LatencyRecorder, now
from util.gpu_check import require_cuda


def load_frames(path, max_frames=None, height=1024, rotate=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        h, w = frame.shape[:2]
        if h != height:
            frame = cv2.resize(frame, (int(round(w * height / h / 2)) * 2, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames, fps


class WarpingError:
    """Accumulates the warping error of an output sequence and of its input sequence over the same region."""

    def __init__(self):
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev = None
        self.err_out, self.err_in, self.n = 0.0, 0.0, 0

    def add(self, frame, output, mask):
        # mask: bool (H, W), where the garment is drawn in this frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is not None and mask is not None and self.prev[3] is not None:
            p_gray, p_frame, p_out, p_mask = self.prev
            flow = self.flow.calc(gray, p_gray, None)  # for each current pixel, where it was in the previous frame
            h, w = gray.shape
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            mx, my = xs + flow[..., 0], ys + flow[..., 1]
            warp = lambda img: cv2.remap(img, mx, my, cv2.INTER_LINEAR)
            region = mask & (warp(p_mask.astype(np.uint8)) > 0)
            if region.sum() > 100:
                diff = lambda a, b: np.abs(a.astype(np.float32) - warp(b).astype(np.float32)).mean(2)[region].mean()
                self.err_out += diff(output, p_out)
                self.err_in += diff(frame, p_frame)
                self.n += 1
        self.prev = (gray, frame, output, mask)

    def result(self):
        n = max(self.n, 1)
        return self.err_out / n, self.err_in / n, self.n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', required=True, help='input video (portrait; a landscape recording needs --rotate)')
    parser.add_argument('--garment', required=True, help='checkpoint name inside --ckpt_dir')
    parser.add_argument('--ckpt_dir', default='./rtv_ckpts')
    parser.add_argument('--out', default=None, help='write the try-on result to this video file')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--rotate', default='none', choices=['cw', 'ccw', '180', 'none'])
    parser.add_argument('--still', type=int, default=None, metavar='FRAME',
                        help='jitter test: repeat this one frame of the video with fresh camera noise each time, as '
                             'if the person stood perfectly still. Any change in the output is then jitter.')
    parser.add_argument('--still_frames', type=int, default=150, help='length of the --still sequence')
    parser.add_argument('--latency_csv', default=None, help='write per-frame stage timings to this CSV')
    add_pipeline_arguments(parser)
    args = parser.parse_args()
    require_cuda()
    rotations = {'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE, '180': cv2.ROTATE_180,
                 'none': None}
    frames, fps = load_frames(args.video, args.max_frames, rotate=rotations[args.rotate])
    if args.still is not None:
        rng = np.random.default_rng(0)
        base = frames[args.still].astype(np.float32)
        # sensor noise (sigma 3 grey levels) and a slight exposure flicker, like a webcam in indoor light
        frames = [np.clip(base * rng.normal(1.0, 0.004) + rng.normal(0.0, 3.0, base.shape), 0, 255).astype(np.uint8)
                  for _ in range(args.still_frames)]
    print('%d frames of %dx%d at %.2f fps' % (len(frames), frames[0].shape[1], frames[0].shape[0], fps))

    processor = FrameProcessor([args.garment], ckpt_dir=args.ckpt_dir, **processor_options(args))
    processor.switch_to_target_garment(0)
    for i in range(10):  # warm-up on the first frame: cuDNN and OpenGL set-up must not count as frame time
        processor(frames[0].copy(), t=-1.0 + i / fps)
    processor._forget_person()
    processor.smpl_regressor.reset_person()

    lat = LatencyRecorder(enabled=True, sync_cuda=True, csv_path=args.latency_csv, window=len(frames),
                          report_every=10 ** 9)
    warping = WarpingError()
    outputs = []
    prev_layer, jitter, edge_jitter, n_jitter = None, 0.0, 0.0, 0
    for i, frame in enumerate(frames):
        ft = lat.new_frame(now())
        out = processor(frame.copy(), ft, t=i / fps)
        lat.finish(ft)
        drawn = ft.info.get('outcome') == 'ok'
        mask = (processor.last_good[1] > 0.5).cpu().numpy() if drawn and processor.last_good is not None else None
        warping.add(frame, out, mask)
        if args.still is not None and drawn:
            # the garment layer alone (colour x alpha), free of the camera noise under it
            garment, alpha = processor.last_good[0], processor.last_good[1].clamp(0, 1)
            layer = torch.cat([garment * alpha[..., None], alpha[..., None] * 255.0], -1)
            if prev_layer is not None:
                region = (layer[..., 3] > 25) | (prev_layer[..., 3] > 25)
                jitter += (layer[..., :3] - prev_layer[..., :3]).abs().mean(-1)[region].mean().item()
                edge_jitter += (layer[..., 3] - prev_layer[..., 3]).abs()[region].mean().item()
                n_jitter += 1
            prev_layer = layer
        if args.out:
            outputs.append(out)
    print(lat.summary())
    lat.close()
    err_out, err_in, n = warping.result()
    print('warping error over %d frame pairs: output %.2f, camera frames %.2f, flicker added by the try-on %.2f'
          % (n, err_out, err_in, err_out - err_in))
    if n_jitter:
        print('still-person jitter over %d frame pairs (grey levels, lower is better): garment colour %.2f, '
              'garment outline (alpha) %.2f' % (n_jitter, jitter / n_jitter, edge_jitter / n_jitter))
    if args.still is None:
        # frame rate of the pipeline alone, without the per-stage GPU synchronisation used for the timings above
        n = min(len(frames), 200)
        torch.cuda.synchronize()
        t0 = now()
        for i in range(n):
            processor(frames[i].copy(), t=1e4 + i / fps)
        torch.cuda.synchronize()
        print('pipeline frame rate without timing overhead: %.1f fps' % (n / (now() - t0)))
    print('peak GPU memory: %.0f MB' % (torch.cuda.max_memory_allocated() / 2 ** 20))
    if args.out:
        import imageio
        writer = imageio.get_writer(args.out, fps=fps, codec='libx264', quality=8)
        for out in outputs:
            writer.append_data(out[:, :, ::-1])
        writer.close()
        print('wrote', args.out)


if __name__ == '__main__':
    main()
