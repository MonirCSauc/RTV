"""Live try-on of the full-body garments (the paper's recurrent networks), wearing the first one at start.

This is rtl_demo.py with a different default garment list: both programs run the same pipeline
(VITON/frame_processor.py) and accept the same options, and either can show any garment, e.g.

    python demo.py --garments lab_03_vmsdp2ta          # an upper-body garment in this program
    python rtl_demo.py --select 7                      # the coat in rtl_demo.py

The camera is chosen as in rtl_demo.py (--camera, see README).
"""
from rtl_demo import main, FULLBODY_GARMENTS, UPPER_GARMENTS

if __name__ == "__main__":
    main(default_garments=FULLBODY_GARMENTS + UPPER_GARMENTS, default_select=1)
