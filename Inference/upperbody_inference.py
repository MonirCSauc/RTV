import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from VITON.viton_upperbody import FrameProcessor
from util.latency import LatencyRecorder, now
from util.gpu_check import require_cuda

def process_video(video_path, garment_name, ckpt_dir='./checkpoints/', latency_csv=None):
    video_loader = MultithreadVideoLoader(video_path,max_height=1024)
    video_writer = MultithreadVideoWriter(outvid='./output.mp4',fps=video_loader.get_fps())
    frame_processor = FrameProcessor([garment_name,],ckpt_dir=ckpt_dir)
    frame_processor.switch_to_target_garment(0)
    lat = LatencyRecorder(enabled=latency_csv is not None, sync_cuda=True, csv_path=latency_csv)
    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        ft = lat.new_frame(now())
        result = frame_processor(frame, ft)
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
    process_video(video_path, garment_name, opt.ckpt_dir, opt.latency_csv)


