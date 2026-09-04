from PyQt5 import QtCore
from queue import Queue
import cv2
import time
import os
import numpy as np
from collections import deque
from ..config import Config
from ...utils.tools import check_save_dir
from ...model.camera import Camera
from ...model.base_result import Media, ViolanceWorkEvent
from ...model.violance_result import ViolanceWarningResult
class SequenceThread(QtCore.QThread):
    sig_is_violence = QtCore.pyqtSignal(bool)
    signal_violance = QtCore.pyqtSignal(ViolanceWarningResult)
    
    def __init__(self, in_buffer: Queue, sequence_buffer: Queue, information_buffer: Queue, camera: Camera):
        super().__init__()
        self._thread_active = False
        self._in_buffer = in_buffer
        self._sequence_buffer = sequence_buffer
        self._output_buffer = Queue()
        self._information_buffer = information_buffer
        self._buffer_size = 1
        self.sequence_length = 16  
        self._classes = ["CO DANH NHAU", ""]
        self._skip = 2
        self.__camera = camera
        self.start_push_event_time = time.time()
        self.config = Config()
    @property
    def thread_active(self):
        return self._thread_active

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, sequence_length):
        self._sequence_length = sequence_length

    @property
    def out_buffer(self):
        return self._output_buffer

    def run(self):
        self._thread_active = True
        print("Thread Sequence Running!")

        frames = []
        count = 0
        is_violence = False

        recent_preds = deque(maxlen=64)  # sliding window 25 frame
        violence_confirmed = False
        last_emit_time = 0
        cooldown_seconds = 10  # không emit lại quá sớm

        while self._thread_active:
            if self._in_buffer.qsize() == 0:
                QtCore.QThread.msleep(1)
                continue

            if self._information_buffer.qsize() > 0:
                is_violence = self._information_buffer.get()

            frame = self._in_buffer.get()
            count += 1

            if self._output_buffer.qsize() < self._buffer_size:
                frame_copy = frame.copy()

                # Thêm kết quả mới vào sliding window
                recent_preds.append(is_violence)
                num_violence_frames = sum(recent_preds)

                # Kiểm tra có nên emit không
                can_emit = (
                    num_violence_frames >= 48  and
                    not violence_confirmed and
                    (time.time() - last_emit_time >= cooldown_seconds)
                )

                if can_emit:
                    violence_confirmed = True
                    last_emit_time = time.time()
                    self.sig_is_violence.emit(True)

                    information = self._classes[0]
                    color = (0, 0, 255)
                    cv2.putText(frame_copy, information, (20, 100),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
                    overlay = frame_copy.copy()
                    red_tint = (0, 0, 255)
                    cv2.rectangle(overlay, (0, 0), (frame_copy.shape[1], frame_copy.shape[0]), red_tint, thickness=-1)
                    alpha = 0.2
                    frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)
                    # Push event
                    rs = ViolanceWarningResult()
                    rs.event_type_id = self.__camera.event_type_id
                    rs.area_id = self.__camera.area_id
                    rs.comp_id = self.__camera.comp_id
                    rs.device_id = self.__camera.id

                    list_media = []
                    save_image_dir = check_save_dir(self.config.ROOT_SAVE_IMAGE_DIR)
                    image_path = os.path.join(save_image_dir, rs.event_id + ".jpg")
                    cv2.imwrite(image_path, frame_copy)

                    media1 = Media()
                    violance_work_event1 = ViolanceWorkEvent()
                    media1.fileName = f"{rs.event_id}.jpg"
                    media1.filePath = os.path.relpath(image_path,self.config.ROOT_SAVE_DIR)
                    media1.fileType = "1"
                    list_media.append(media1)

                    rs.list_media = list_media
                    rs.violance_work_event = violance_work_event1.serialize()
                    self.signal_violance.emit(rs)
                    self.start_push_event_time = time.time() + self.config.TIME_TO_PUSH_EVENT
                else:
                    if not is_violence:
                        violence_confirmed = False
                    label = self._classes[0] if is_violence else self._classes[1]
                    color = (0, 0, 255) if is_violence else (0, 255, 0)
                    cv2.putText(frame_copy, label, (20, 100),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)

                self._output_buffer.put(frame_copy)

            # Xử lý chuỗi frame để đưa vào sequence
            if count % self._skip != 1:
                continue

            frames.append(frame)
            if len(frames) == self._sequence_length:
                if self._sequence_buffer.qsize() < self._buffer_size:
                    self._sequence_buffer.put(frames)
                frames = []

            QtCore.QThread.msleep(1)

    def stop(self):
        self._thread_active = False
        print("Stop Sequence Thread")

    
    def stop(self):
        self._thread_active = False
        print("Stop Sequence Thread")
