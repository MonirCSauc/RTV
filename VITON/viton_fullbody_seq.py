from VITON.frame_processor import FrameProcessor


class FullBodySeqFrameProcessor(FrameProcessor):
    """The full-body recurrent pipeline (paper's ReGarSyn) with the earlier interface: one garment, .forward(frame).

    It is the shared pipeline of VITON/frame_processor.py; the garment's kind selects the 576x432 full-body crop,
    the full-body VM template, the SSDP body map and the recurrent network.
    """

    def __init__(self, target_name='coat_seq_vmssdp2ta_576', ckpt_dir='./rtv_ckpts', **options):
        super().__init__([target_name], ckpt_dir=ckpt_dir, **options)
        self.switch_to_target_garment(0)

    def forward(self, raw_image, ft=None, t=None):
        return self(raw_image, t=t) if ft is None else self(raw_image, ft, t)
