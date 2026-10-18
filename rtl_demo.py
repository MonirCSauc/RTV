import sys
import os
import threading
import argparse

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


DEFAULT_GARMENTS = ['lab_03_vmsdp2ta', 'lab_04_vmsdp2ta', 'lab_07_vmsdp2ta',
                    'jin_17_vmsdp2ta', 'jin_18_vmsdp2ta', 'jin_22_vmsdp2ta']
ROTATIONS = {'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE, '180': cv2.ROTATE_180, 'none': None}


class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage, object)  # (image, FrameTiming)

    def __init__(self,garment_id_list,camera=None,ckpt_dir=None,rotate='cw',latency_csv=None,marker=False):
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
        self.grabber = None

    def set_taregt_id(self, id):
        print(id)
        self.frame_processor.set_target_garment(id)

    def get_camera(self, camera=None):
        # --camera / RTV_CAMERA: camera index or stream URL (e.g. http://192.168.1.5:4747/video for DroidCam)
        src = camera or os.environ.get('RTV_CAMERA')
        if src and not src.isdigit():
            # a stream URL; its resolution is set on the sending side (e.g. the DroidCam phone app)
            return cv2.VideoCapture(src)
        if src:
            return self._request_hd(cv2.VideoCapture(int(src)))
        # Try each index/backend and keep the first one that actually delivers a frame
        # (isOpened() alone is true for virtual/busy cameras that never produce video).
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == 'win32' else [cv2.CAP_ANY]
        for index in (1, 0, 2, 3):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    self._request_hd(cap)
                    for _ in range(10):
                        if cap.read()[0]:
                            print(f"Using camera {index} (backend {backend})")
                            return cap
                cap.release()
        print("No working camera found. Pass --camera (or set RTV_CAMERA) to an index or stream URL.")
        return cv2.VideoCapture()

    @staticmethod
    def _request_hd(cap):
        # OpenCV opens most webcams at 640x480 unless asked; request 1080p (the camera picks the closest it supports)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return cap

    def _grab_latest(self):
        # Drain the stream continuously so processing never works on stale, buffered frames
        while self.running:
            ret, frame = self.cap.read()
            t_cap = now()  # earliest software timestamp; exposure and transfer delay happen before this
            if ret:
                self.lat.on_capture()
                self._latest = (frame, t_cap)
            else:
                self.msleep(10)

    def run(self):
        self._latest = None
        self.grabber = threading.Thread(target=self._grab_latest, daemon=True)
        self.grabber.start()
        # RTV_HEIGHT: height the try-on is composited at. Frames are scaled up to it so the garment stays
        # sharp on a low-resolution camera; lower it for more speed.
        max_h = int(os.environ.get('RTV_HEIGHT', '1024'))
        max_w = max_h * 9 // 16
        while self.running:
            item, self._latest = self._latest, None
            if item is None:
                self.msleep(2)
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
                frame=cv2.resize(frame, (max_w, max_h),
                                 interpolation=cv2.INTER_CUBIC if frame.shape[0] < max_h else cv2.INTER_AREA)
            frame = self.frame_processor(frame, ft)
            with ft.stage('qimage'):
                if self.marker:
                    frame = draw_latency_marker(frame.copy(), ft.frame_id, t_cap)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                # copy: the QImage is delivered to the GUI thread later, after this frame's buffer is gone
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888).copy()
            ft.mark('emit')
            self.frameCaptured.emit(q_img, ft)

    def reset_background(self):
        self.frame_processor.reset_background()

    def toggle_remove_worn(self):
        self.frame_processor.remove_worn = not self.frame_processor.remove_worn
        print("Remove worn garment:", self.frame_processor.remove_worn)

    def stop(self):
        self.running = False
        if self.grabber is not None:
            self.grabber.join(timeout=1.0)
        self.cap.release()
        if self.lat.enabled:
            print(self.lat.summary())
        self.lat.close()

class CameraApp(QMainWindow):
    def __init__(self, garment_name_list, camera=None, ckpt_dir=None, rotate='cw', latency_csv=None, marker=False):
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
        self.viton_thread = VitonThread(garment_name_list, camera, ckpt_dir, rotate, latency_csv, marker)
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress):
            # print(f'keypress obj: {obj}')
            if obj == self.list_widget:
                self.list_widget.setCurrentRow(event.key()-48)
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
        elif e.key() == Qt.Key_B:
            # relearn the background plate used to erase the worn garment
            self.viton_thread.reset_background()
        elif e.key() == Qt.Key_R:
            self.viton_thread.toggle_remove_worn()
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
                        help='camera index or stream URL; defaults to RTV_CAMERA, then the first working device')
    parser.add_argument('--garments', default=','.join(DEFAULT_GARMENTS),
                        help='comma-separated checkpoint names inside --ckpt_dir')
    parser.add_argument('--ckpt_dir', default='./rtv_ckpts', help='directory holding the garment checkpoints')
    parser.add_argument('--rotate', default='cw', choices=list(ROTATIONS),
                        help='rotation applied to camera frames; cw suits a camera mounted sideways')
    parser.add_argument('--latency_csv', default=None,
                        help='write per-frame stage timings and capture-to-display latency to this CSV')
    parser.add_argument('--latency_marker', action='store_true',
                        help='draw the frame id and capture time into the output, for filming the mirror')
    args, qt_args = parser.parse_known_args()
    require_cuda()
    app = QApplication(sys.argv[:1] + qt_args)
    window = CameraApp(args.garments.split(','), args.camera, args.ckpt_dir, args.rotate,
                       args.latency_csv, args.latency_marker)
    window.show()
    sys.exit(app.exec_())
