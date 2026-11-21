# Real-Time Virtual Try-On

[![arXiv](https://img.shields.io/badge/arXiv-2506.12348-b31b1b.svg)](https://arxiv.org/abs/2506.12348)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/wuzaiqiang/rtv_ckpts)

This project supports real-time virtual try-on without the need for any special sensors. All you need is a webcam and a PC with a GPU equivalent to an RTX 3060 or higher.

Unlike other image-based virtual try-on methods, our approach requires training a dedicated network for each garment in a low-barrier, accessible way (see [this paper](https://arxiv.org/abs/2506.10468)).
Our method simply overlay the synthesized garment on the top of human body without removing the original garment to achieve real-time performace.

<p align="center">
  <img src="assets/output.gif" width="45%" />
  <img src="assets/Han.gif" width="46%" />
</p>


Check our [tight garment demo](https://www.youtube.com/watch?v=dub0pJ2kXA4) and [loose garment demo](https://www.youtube.com/watch?v=7hm1yBsFzHc) on YouTube.


## Installation

```
git clone https://github.com/ZaiqiangWu/RTV.git
cd RTV
```

### Environment
```
sudo apt install gcc g++ libxcb-xinerama0-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xkb-dev 
sudo apt install qtbase5-dev qtbase5-dev-tools libqt5gui5 libqt5widgets5 libqt5multimedia5 libqt5multimediawidgets5 libqt5multimedia5-plugins libpulse-mainloop-glib0
conda create -n rtv python=3.12
conda activate rtv
pip install -r requirements.txt
pip install --no-build-isolation detectron2@git+https://github.com/facebookresearch/detectron2.git
pip install --no-build-isolation git+https://github.com/ZaiqiangWu/ROMP.git#subdirectory=simple_romp
```

### Weights
Download our pretrained checkpoints by running the following command. Please refer to [this page](./training_instructions.md) for instructions on training your own garment items.
```
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/wuzaiqiang/rtv_ckpts
```

### Real-time virtual try-on
Please make sure that there is a webcam connected to your PC before running the following command.
```
python rtl_demo.py
```
Using a phone as the camera (DroidCam):
- Over Wi-Fi, no PC client needed: `python rtl_demo.py --camera http://<phone-ip>:4747/video` (the default is `http://10.40.33.34:4747/video`; `--camera auto` picks the first local camera instead) (the IP and port are shown in the DroidCam app; the phone and PC must be on the same network, and the DroidCam PC client must be closed, since the phone serves one connection at a time).
- With the DroidCam PC client (Wi-Fi or USB): connect in the client, then run `python util/camera_util.py` to find its camera number and pass `--camera <number>`. On Windows the demo opens cameras through DirectShow first, which DroidCam's virtual camera needs.
- Hold the phone upright on a stand; the default `--rotate cw` turns DroidCam's sideways image upright (use `--rotate ccw` if you appear upside down). The free DroidCam app sends at most 640x480; frames are scaled to 576x1024 for the models.

On Windows the body renderer uses a hidden GLFW window for OpenGL (`pip install glfw`); on Linux it uses EGL.

Useful options:
- `--camera <index or URL>`: pick the camera, e.g. `--camera 2` or a phone stream such as `--camera http://<phone-ip>:4747/video` (DroidCam).
- `--rotate {cw,ccw,180,none}`: rotation applied to camera frames (default `cw`, for a camera mounted sideways in portrait).
- `--garments <names>` and `--ckpt_dir <dir>`: comma-separated checkpoint names to offer, and the folder holding them (default `./rtv_ckpts`). Upper-body garments (`*_vmsdp2ta`) and full-body garments (`*_seq_vmssdp2ta_576`) can be mixed; garments whose weights are missing are left out with a message.
- `--select <n>`: wear garment number `n` of the list at start. Keys `1`-`9` pick a garment, `0` none.
- `--camera <video file>` plays a recorded video in real time and repeats it, to test without a camera (add `--rotate none` for a portrait video).
- `--latency_csv <file>`: record each stage's time on the GPU and the delay from camera capture to display, one CSV row per frame, with a p50/p95 summary printed every 100 frames and on exit. Timing synchronises the GPU after each stage, which costs a little speed; leave it off when measuring top fps.
- `--latency_marker`: draw the frame number and capture time into the output, so a high-speed phone video of the mirror can measure glass-to-glass delay.

- `--debug`: show, next to the output, the camera input, the rendered 3D body (VM), the DensePose body map (SDP), the garment colour the network produced, and its mask (square-root scaled so faint values show) with the hands and face kept in front of the garment in red. It also warns if any stage draws into the camera frame. Press `S` to save the displayed frame and the full-frame arrays (`.png` + `.npz`) to `./debug_frames/`.

- `--dp_score <0..1>`: minimum DensePose detection score (default 0.5). DensePose always describes the same person as the 3D body (BEV). The latency CSV records per frame how many people each model saw, how well they agreed (`dp_joint_fraction`), and the frame `outcome` (`ok`, `no_person`, `dp_mismatch`, `dp_no_person`, `no_garment`).
- `--track {on,off}` (default on): the garment stays on one customer. A new customer is the nearest person inside `--zone`; after that the same person is followed by position and apparent size, so someone walking by, even in front, does not take the garment. If the customer is hidden for 1.5 s the mirror takes the next person. `off` takes the nearest person in every frame (original).
- `--zone L,R` (default `0,1`): fractions of the image width in which a new customer is accepted, e.g. `0.2,0.8` ignores people at the sides.
- `--smooth {on,off}` (default on): steadies the body estimate (pose, position, shape), the crop and the DensePose box over time with filters that smooth strongly at rest and hardly at all in fast motion. `off` is the original behaviour.
- `--temporal <0..1>` (default 0.5): flicker filter on the output of the upper-body (per-frame) networks; 0 turns it off. The full-body networks have their own memory and are not filtered.
- `--dp_mode {box,full}` (default box): `box` runs DensePose only inside the body box BEV found (no person search; about 15 ms instead of 40 ms); `full` runs the whole detector (original).
- `--fp16 {on,off}` (default on): half precision for DensePose and the garment network.
- `--cover_worn {on,off}` (default on): stops the customer's own top from showing around the new garment. The garment grows over adjacent pixels that have the colour of the customer's top (strips on the shoulders, sleeve ends, the collar in the neck opening), using its own colour next to them; small holes in it are closed; and the head and hand masks no longer keep the customer's collar or the cloth around a fist in front of the garment. Nothing is painted on the customer and no training is involved. The grown part carries the garment's colour but not its texture, so it reaches at most 14% of the garment's height; a much larger or longer top still shows beyond that. It does nothing when the top and the customer's skin have about the same colour.
- `--original`: all of the above as in the original pipeline, for comparisons.

- `--body_shape {estimated,train,zero}`: body shape of the rendered 3D body. `estimated` (default for upper-body garments) follows the customer's own shape; full-body garments default to `train`; `train` (average shape, as in the training data) and `zero` are there for experiments and can make the garment narrower or wider than the person.
- `--keep_in_front {hands_head,hands,none}`: body parts drawn in front of the garment (default hands and head). Seen from behind, DensePose's head is the back of the head and hair, which can cut into a garment's back collar; `hands` keeps only the hands in front.
- `--blend {soft,original}`: garment edges. `soft` (default) blends anti-aliased edges without a dark outline; `original` uses the repository's original hard 50% mask. `--blend original --keep_in_front none` draws exactly like the original code.
- `--alpha_gate PX` (default 30): keeps the garment on the body. A garment network shown a pose it never saw in training (e.g. `lab_03` with the left arm raised overhead) can spread alpha over the whole crop, which shows as a dark, blocky ghost on the wall. The gate keeps only confident garment regions that touch the 3D body render, at most PX pixels (in the 512 crop) beyond the body. On the poses the networks handle it removes nothing visible. In `--debug`, removed alpha shows in blue. `0` turns it off.
- `--keep_garments_on_gpu`: keep every garment network on the GPU (~0.8 GB each) so switching is instant. Only with plenty of GPU memory: on Windows, running short does not fail but silently slows every frame.

- `--lock_camera`: let the camera adjust for about a second, then freeze its exposure and white balance, so changing store light or people walking by do not make the image (and the body estimates) flicker. Works for USB and virtual cameras whose driver supports it (the result is printed); for a phone stream, lock exposure in the phone app.

`python demo.py` is the same program with the full-body garments first and the coat worn at start; both programs take the same options and can show any garment (e.g. `python demo.py --garments lab_03_vmsdp2ta`, `python rtl_demo.py --select 7`).

### Measuring speed and stability
```
python tools/benchmark_video.py --video <video> --garment lab_03_vmsdp2ta            # new pipeline
python tools/benchmark_video.py --video <video> --garment lab_03_vmsdp2ta --original # original pipeline
python tools/benchmark_video.py --video <video> --garment lab_03_vmsdp2ta --still 0  # jitter test on frame 0
```
It prints per-stage times, the frame rate, the warping error (flicker beyond the person's real motion) and, with `--still`, the jitter of the garment for a person standing perfectly still (one frame repeated with fresh camera noise). `--out result.mp4` saves the result.

At start-up the demo checks that PyTorch can see a CUDA GPU and prints where BEV, DensePose and each garment network run (they should all show `cuda:0`).

## BibTeX
```text
@inproceedings{wu2025real,
  title={Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments},
  author={Wu, Zaiqiang and Shen, I-Chao and Igarashi, Takeo},
  booktitle={Computer Graphics Forum},
  pages={e70272},
  year={2025},
  organization={Wiley Online Library}
}

@article{wu2025low,
  title={Low-barrier dataset collection with real human body for interactive per-garment virtual try-on},
  author={Wu, Zaiqiang and Li, Yechen and Liu, Jingyuan and Shibata, Yuki and Hori, Takayuki and Shen, I-Chao and Igarashi, Takeo},
  journal={IEEE Computer Graphics and Applications},
  year={2025},
  publisher={IEEE}
}
```

## License
This project is licensed under a custom Apache 2.0-based license with a **Non-Commercial Use** clause.  
You may use, modify, and share the code freely for non-commercial purposes.  
Commercial use requires prior written permission from the author.  
See the [LICENSE](./LICENSE) file for details.
