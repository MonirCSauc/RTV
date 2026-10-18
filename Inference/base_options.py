import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input_video', type=str, help='the path of input video file')
        self.parser.add_argument('--garment_name', type=str, default='example_garment', help='id of the target garment')
        self.parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='folder holding the garment checkpoints')
        self.parser.add_argument('--latency_csv', type=str, default=None, help='write per-frame stage timings to this CSV')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


