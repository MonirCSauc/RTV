import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from VITON.frame_processor import FrameProcessor
from VITON.options import processor_options
from util.latency import LatencyRecorder, now
from util.gpu_check import require_cuda

def process_video(video_path, garment_name, ckpt_dir='./checkpoints/', latency_csv=None, output='./output.mp4',
                  options=None):
    # works for upper-body and full-body garments alike (the garment's kind selects the pipeline settings)
    video_loader = MultithreadVideoLoader(video_path,max_height=1024)
    fps = video_loader.get_fps() or 30.0
    video_writer = MultithreadVideoWriter(outvid=output,fps=fps)
    frame_processor = FrameProcessor([garment_name,],ckpt_dir=ckpt_dir, **(options or {}))
    frame_processor.switch_to_target_garment(0)
    lat = LatencyRecorder(enabled=latency_csv is not None, sync_cuda=True, csv_path=latency_csv)
    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        if frame is None:
            break
        ft = lat.new_frame(now())
        # the time of a frame is its place in the video, not how long processing took
        result = frame_processor(frame, ft, t=i / fps)
        lat.finish(ft)
        video_writer.append(result)
    video_writer.make_video()
    video_writer.close()
    if lat.enabled:
        print(lat.summary())
    lat.close()

if __name__ == '__main__':
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.input_video
    garment_name = opt.garment_name
    require_cuda()
    process_video(video_path, garment_name, opt.ckpt_dir, opt.latency_csv, opt.output, processor_options(opt))


