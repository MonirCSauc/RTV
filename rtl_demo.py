import sys
import os
import argparse
import threading
import time

#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QUrl, QEvent, QFileInfo
# from PyQt5 import QtMultimedia
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from util.image_warp import crop2_169

from VITON.viton_upperbody import FrameProcessor
from util.camera_util import list_available_cameras
from util.latency import LatencyRecorder, now, draw_latency_marker
from util.gpu_check import require_cuda
from util.debug_view import compose_debug_panel, save_debug


DEFAULT_GARMENTS = ['lab_03_vmsdp2ta', 'lab_04_vmsdp2ta', 'lab_07_vmsdp2ta',
                    'jin_17_vmsdp2ta', 'jin_18_vmsdp2ta', 'jin_22_vmsdp2ta']
ROTATIONS = {'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE, '180': cv2.ROTATE_180, 'none': None}


class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage, object)  # (image, FrameTiming)

    def __init__(self,garment_id_list,camera=None,ckpt_dir=None,rotate='cw',latency_csv=None,marker=False,debug=False):
        super().__init__()
        self.cap = self.get_camera(camera)
        if not self.cap.isOpened():
            print("Failed to open the selected camera.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera resolution: %dx%d" % (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.running = True
        self.rotate = ROTATIONS[rotate]
        # timing is off unless --latency_csv is given; then every stage syncs CUDA so its GPU time is billed correctly
        self.lat = LatencyRecorder(enabled=latency_csv is not None, sync_cuda=True, csv_path=latency_csv)
        self.marker = marker
        self.frame_processor = FrameProcessor(garment_id_list, ckpt_dir=ckpt_dir)
        self.frame_processor.debug = debug
        self.save_next = False  # set by the S key: save the next displayed frame to ./debug_frames/
        # keep only the newest camera frame, so slow processing never shows stale (laggy, smeared) frames
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.grabber = threading.Thread(target=self.grab_frames, daemon=True)
        self.grabber.start()

    def set_taregt_id(self, id):
        print(id)
        self.frame_processor.set_target_garment(id)

    def get_camera(self, camera=None):
        # --camera / RTV_CAMERA: a camera index, or a stream URL such as DroidCam's http://<phone-ip>:4747/video
        src = camera or os.environ.get('RTV_CAMERA')
        if src and not src.isdigit():
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                print("Could not open the stream %s; check the phone's IP and port and that both are on the same "
                      "network (open the URL in a browser to test)." % src)
            return cap
        # Windows: the DroidCam client installs a DirectShow device, which Media Foundation (OpenCV's default)
        # often cannot open, so try DirectShow first
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == 'win32' else [cv2.CAP_ANY]
        for index in ([int(src)] if src else [1, 0, 2, 3]):
            for backend in backends:
                cap = self._open_delivering(index, backend)
                if cap is not None:
                    return cap
        print("No working camera found. Pass --camera (or set RTV_CAMERA) to a camera index or stream URL.")
        return cv2.VideoCapture()

    @staticmethod
    def _delivers_frames(cap, tries=20):
        # isOpened() is also true for virtual or busy cameras that never produce video
        return any(cap.read()[0] for _ in range(tries))

    def _open_delivering(self, index, backend):
        for request_hd in (True, False):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                return None
            if request_hd:
                # OpenCV opens most webcams at 640x480 unless asked; request 1080p (the camera picks the closest)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            if self._delivers_frames(cap):
                print("Using camera %d (OpenCV backend %s)%s" % (index, cap.getBackendName(),
                                                                 '' if request_hd else ', at its default resolution'))
                return cap
            # some virtual cameras (e.g. DroidCam) stop delivering when asked for MJPG/1080p: retry at defaults
            cap.release()
        return None

    def grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            t_cap = now()  # earliest software timestamp; exposure and transfer delay happen before this
            if not ret:
                time.sleep(0.01)
                continue
            self.lat.on_capture()
            with self.frame_lock:
                self.latest_frame = (frame, t_cap)

    def run(self):
        while self.running:
            with self.frame_lock:
                item, self.latest_frame = self.latest_frame, None
            if item is None:
                time.sleep(0.005)
                continue
            frame, t_cap = item
            ft = self.lat.new_frame(t_cap)
            ft.mark('dequeued')  # dequeued - captured = time spent waiting for the previous frame to finish

            with ft.stage('prep'):
                # the camera is usually mounted sideways (portrait); see --rotate
                if self.rotate is not None:
                    frame = cv2.rotate(frame, self.rotate)
                ## ichao: remove flip (Nov 13, 2024)
                frame=cv2.flip(frame, 1)
                frame=crop2_169(frame)
                # one working size for every camera (models and training data use 1024-high frames)
                frame=cv2.resize(frame, (576, 1024),
                                 interpolation=cv2.INTER_CUBIC if frame.shape[0] < 1024 else cv2.INTER_AREA)
            frame = self.frame_processor(frame, ft)
            if self.frame_processor.debug and self.frame_processor.last_debug is not None:
                frame = compose_debug_panel(frame, self.frame_processor.last_debug)
            if self.save_next:
                self.save_next = False
                path = os.path.join('debug_frames', time.strftime('%Y%m%d_%H%M%S') + '_%d.png' % ft.frame_id)
                save_debug(path, frame, self.frame_processor.last_debug)
                print('saved', path, '(+ .npz with the full-frame arrays)' if self.frame_processor.last_debug else '')
            with ft.stage('qimage'):
                if self.marker:
                    frame = draw_latency_marker(frame.copy(), ft.frame_id, t_cap)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                # copy: the QImage must own its pixels, since the numpy buffer is freed before the UI draws it
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888).copy()
            ft.mark('emit')
            self.frameCaptured.emit(q_img, ft)


    def stop(self):
        self.running = False
        self.grabber.join(timeout=1.0)
        self.cap.release()
        if self.lat.enabled:
            print(self.lat.summary())
        self.lat.close()

class CameraApp(QMainWindow):
    def __init__(self, garment_name_list, camera=None, ckpt_dir=None, rotate='cw', latency_csv=None, marker=False,
                 debug=False):
        super().__init__()
        self.setWindowTitle("Virtual Try-On")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(200, 150)
        ## enable fullscreen
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)

        # Create a scroll area for the horizontal layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create a QWidget to hold the list
        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)

        # Create a QListWidget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)  # Ensure only one item can be selected at a time
        self.list_widget.setIconSize(QSize(150,150))

        # Add none item
        item = QListWidgetItem()
        item.setIcon(QIcon('./assets/none.png'))
        item.setData(Qt.UserRole, -1)
        self.list_widget.addItem(item)

        # Add 15 images to the horizontal layout
        #garment_id_list = [3, 2, 17, 18, 22]
        
        for i, garment in enumerate(garment_name_list):
            item = QListWidgetItem()
            icon_path = './assets/garment_images/'+garment.split('_vmsdp2ta')[0]+'_white_bg.jpg'
            if os.path.exists(icon_path):
                item.setIcon(QIcon(icon_path))
            else:
                item.setText(garment)
            item.setData(Qt.UserRole, i)
            self.list_widget.addItem(item)
        #for i in range(9):
        #    item = QListWidgetItem()
        #    item.setIcon(QIcon('./figures/female_garments/lab_'+str(i).zfill(2)+'_white_bg.jpg'))
        #    item.setData(Qt.UserRole, 16+i)
        #    self.list_widget.addItem(item)
        layout_list.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)


        
        scroll_area.setWidget(container_list)

        layout.addWidget(scroll_area)
        scroll_area.setFixedWidth(230)

        #layout.setStretch(0, 9)
        #layout.setStretch(1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.viton_thread = VitonThread(garment_name_list, camera, ckpt_dir, rotate, latency_csv, marker, debug)
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress):
            # print(f'keypress obj: {obj}')
            if obj == self.list_widget:
                if Qt.Key_0 <= event.key() <= Qt.Key_9:
                    self.list_widget.setCurrentRow(event.key()-48)
                else:
                    self.keyPressEvent(event)  # S (save) and Q/Esc (quit) also work while the list has focus
                return True
        
        return super(CameraApp, self).eventFilter(obj, event)
            
        
                
    def keyPressEvent(self, e):
        print(f'key: {e.key()}')
        if e.key() == Qt.Key_0:
            self.list_widget.setCurrentRow(0)
        elif e.key() == Qt.Key_1:
            self.list_widget.setCurrentRow(1)
        elif e.key() == Qt.Key_2:
            self.list_widget.setCurrentRow(2)
        elif e.key() == Qt.Key_3:
             self.list_widget.setCurrentRow(3)
        elif e.key() == Qt.Key_4:
             self.list_widget.setCurrentRow(4)
        elif e.key() == Qt.Key_5:
             self.list_widget.setCurrentRow(5)
        elif e.key() == Qt.Key_S:
            self.viton_thread.save_next = True
        elif e.key() == Qt.Key_Q or e.key() == Qt.Key_Escape:
            print("Exiting application...")
            self.close()  # Triggers closeEvent

    def on_selection_changed(self):
        print('selection changed')
        selected_items = self.list_widget.selectedItems() 
        ## TODO: test if it can keep playing ...
        if selected_items:
            selected_item = selected_items[0]
            item_id = selected_item.data(Qt.UserRole)  # Retrieve the ID from the item

            ## change garment
            self.viton_thread.set_taregt_id(item_id)
            
            #self.trigger_function(item_id)  # Pass the ID to the trigger function



    def update_image(self, q_img, ft):
        ft.mark('gui_recv')  # delay of the queued signal from the processing thread to the GUI thread
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.viton_thread.lat.enabled:
            self.image_label.repaint()  # paint now, so 'shown' means handed to the window system
            self.viton_thread.lat.finish(ft)

    def closeEvent(self, event):
        self.viton_thread.stop()
        self.viton_thread.wait()
        print("Viton thread stopped")
        event.accept()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=None,
                        help='camera index or stream URL (e.g. http://<phone-ip>:4747/video for DroidCam); '
                             'defaults to RTV_CAMERA, then the first camera that delivers frames')
    parser.add_argument('--garments', default=','.join(DEFAULT_GARMENTS),
                        help='comma-separated checkpoint names inside --ckpt_dir')
    parser.add_argument('--ckpt_dir', default='./rtv_ckpts', help='directory holding the garment checkpoints')
    parser.add_argument('--rotate', default='cw', choices=list(ROTATIONS),
                        help='rotation applied to camera frames; cw suits a camera mounted sideways')
    parser.add_argument('--latency_csv', default=None,
                        help='write per-frame stage timings and capture-to-display latency to this CSV')
    parser.add_argument('--latency_marker', action='store_true',
                        help='draw the frame id and capture time into the output, for filming the mirror')
    parser.add_argument('--debug', action='store_true',
                        help='show what the garment network sees (VM, SDP, alpha) next to the output; S saves a frame')
    args, qt_args = parser.parse_known_args()
    require_cuda()
    app = QApplication(sys.argv[:1] + qt_args)
    window = CameraApp(args.garments.split(','), args.camera, args.ckpt_dir, args.rotate,
                       args.latency_csv, args.latency_marker, args.debug)
    window.show()
    sys.exit(app.exec_())
