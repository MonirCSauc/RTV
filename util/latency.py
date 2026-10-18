"""Per-stage and capture-to-display latency instrumentation for rtl_demo.py.

Usage (see A4_latency_cost.md for the exact patch):
    from util.latency import LatencyRecorder
    rec = LatencyRecorder(enabled=True, sync_cuda=True, csv_path='latency.csv')
    ft = rec.new_frame(t_cap)            # t_cap = perf_counter() taken right after cap.read()
    with ft.stage('bev'): ...            # synchronises CUDA before reading the clock
    ft.mark('emit')                      # an instant (no duration)
    rec.finish(ft)                       # after the frame is painted; logs + rolling stats
All times are time.perf_counter() seconds; reported in ms.
"""
import csv
import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # torch missing or broken: timing still works, without GPU sync
    torch, _HAS_CUDA = None, False


# fixed CSV columns, so frames that skip stages (no garment selected, no person found) still line up
STAGES = ['prep', 'bev', 'vm_render', 'densepose', 'inputs', 'generator', 'warp_back', 'composite', 'qimage']
MARKS = ['dequeued', 'emit', 'gui_recv', 'shown']
# per-frame facts logged alongside the timings (see FrameProcessor): people seen by BEV and by DensePose, the share
# of the BEV person's joints inside the chosen DensePose box (low = the two models disagree), and the frame outcome
INFO = ['n_people_bev', 'n_people_dp', 'dp_joint_fraction', 'outcome']
COLUMNS = (['frame_id', 't_cap'] + ['stage_%s_ms' % s for s in STAGES] + ['at_%s_ms' % m for m in MARKS]
           + ['cap_to_shown_ms'] + INFO)


def now():
    return time.perf_counter()


def cuda_sync():
    if _HAS_CUDA:
        torch.cuda.synchronize()


class FrameTiming:
    """Timing record that travels with one camera frame through the pipeline."""
    __slots__ = ('frame_id', 't_cap', 'stages', 'marks', 'info', '_sync', '_enabled')

    def __init__(self, frame_id, t_cap, sync_cuda, enabled):
        self.frame_id = frame_id
        self.t_cap = t_cap
        self.stages = {}   # name -> duration (s); repeated names accumulate
        self.marks = {}    # name -> absolute perf_counter time (s)
        self.info = {}     # name -> value, see INFO
        self._sync = sync_cuda
        self._enabled = enabled

    @contextmanager
    def stage(self, name):
        if not self._enabled:
            yield
            return
        if self._sync:
            cuda_sync()          # do not bill earlier queued GPU work to this stage
        t0 = now()
        try:
            yield
        finally:
            if self._sync:
                cuda_sync()      # include this stage's queued GPU work
            self.stages[name] = self.stages.get(name, 0.0) + (now() - t0)

    def note(self, **values):
        if self._enabled:
            self.info.update(values)

    def mark(self, name, t=None):
        if self._enabled:
            self.marks[name] = now() if t is None else t


class _NullTiming(FrameTiming):
    def __init__(self):
        super().__init__(-1, 0.0, False, False)


NULL_TIMING = _NullTiming()


class LatencyRecorder:
    """Collects FrameTiming records; thread-safe; prints rolling p50/p95 every `report_every` frames."""

    def __init__(self, enabled=True, sync_cuda=True, csv_path=None, window=300, report_every=100):
        self.enabled = enabled
        self.sync_cuda = sync_cuda and _HAS_CUDA
        self.window = deque(maxlen=window)
        self.report_every = report_every
        self.lock = threading.Lock()
        self.next_id = 0
        self.n_captured = 0          # frames read from the camera
        self.n_finished = 0          # frames that reached the screen
        self.t_first = None
        self.csv_path = csv_path
        self._csv = None
        self._writer = None
        self._closed = False

    # --- capture side -------------------------------------------------------
    def on_capture(self):
        """Call once per cap.read() in the grabber thread (counts frames the newest-frame queue drops)."""
        with self.lock:
            self.n_captured += 1

    def new_frame(self, t_cap):
        if not self.enabled:
            return NULL_TIMING
        with self.lock:
            fid = self.next_id
            self.next_id += 1
            if self.t_first is None:
                self.t_first = t_cap
        return FrameTiming(fid, t_cap, self.sync_cuda, True)

    # --- display side -------------------------------------------------------
    def finish(self, ft, t_shown=None):
        if not self.enabled or ft is NULL_TIMING:
            return
        ft.mark('shown', t_shown)
        row = {'frame_id': ft.frame_id, 't_cap': ft.t_cap}
        row.update({'stage_' + k + '_ms': v * 1e3 for k, v in ft.stages.items()})
        row.update({'at_' + k + '_ms': (t - ft.t_cap) * 1e3 for k, t in ft.marks.items()})
        row['cap_to_shown_ms'] = (ft.marks['shown'] - ft.t_cap) * 1e3
        row.update(ft.info)
        with self.lock:
            if self._closed:  # a frame still in flight while the app shuts down
                return
            self.n_finished += 1
            self.window.append(row)
            self._write(row)
            if self.n_finished % self.report_every == 0:
                print(self.summary())

    def _write(self, row):
        if self.csv_path is None:
            return
        if self._writer is None:
            self._csv = open(self.csv_path, 'w', newline='')
            self._writer = csv.DictWriter(self._csv, fieldnames=COLUMNS, extrasaction='ignore')
            self._writer.writeheader()
        self._writer.writerow(row)
        if self.n_finished % 30 == 0:
            self._csv.flush()

    def summary(self):
        rows = list(self.window)
        if not rows:
            return 'latency: no frames'
        keys = sorted({k for r in rows for k in r if k.endswith('_ms')})
        lines = ['latency over last %d frames (ms)   p50     p95     p99' % len(rows)]
        for k in keys:
            v = np.array([r[k] for r in rows if k in r])
            lines.append('  %-28s %7.1f %7.1f %7.1f' % (k, *np.percentile(v, [50, 95, 99])))
        span = rows[-1]['t_cap'] - rows[0]['t_cap']
        if span > 0:
            lines.append('  displayed fps %.1f; frames captured %d, shown %d (dropped %.0f%%)' % (
                (len(rows) - 1) / span, self.n_captured, self.n_finished,
                100.0 * (1 - self.n_finished / max(self.n_captured, 1))))
        outcomes = [r['outcome'] for r in rows if 'outcome' in r]
        if outcomes:
            names, counts = np.unique(outcomes, return_counts=True)
            lines.append('  frame outcomes: ' + ', '.join('%s %d%%' % (n, round(100.0 * c / len(outcomes)))
                                                          for n, c in zip(names, counts)))
        return '\n'.join(lines)

    def close(self):
        with self.lock:
            self._closed = True
            if self._csv is not None:
                self._csv.close()
                self._csv = None


def draw_latency_marker(frame_bgr, frame_id, t_cap, size=48):
    """Burn a marker into the OUTPUT image for external (phone-camera) measurement.

    Top-left square alternates black/white with the frame id parity, and the capture time (ms, modulo
    100 s) is printed, so a 240 fps phone video of [reference clock/LED + mirror] can be read frame by frame.
    """
    import cv2
    v = 255 if frame_id % 2 else 0
    frame_bgr[:size, :size] = v
    txt = '%d %.0f' % (frame_id, (t_cap * 1e3) % 1e5)
    cv2.putText(frame_bgr, txt, (size + 6, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame_bgr
