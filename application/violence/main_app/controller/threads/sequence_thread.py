from PyQt5 import QtCore
from queue import Queue
import cv2
import time
import os
import numpy as np
import math
import torch
from collections import deque

from ..config import Config
from ...utils.tools import check_save_dir
from ...model.camera import Camera
from ...model.base_result import Media, ViolanceWorkEvent
from ...model.violance_result import ViolanceWarningResult

class SequenceThread(QtCore.QThread):
    sig_is_violence = QtCore.pyqtSignal(bool)
    signal_violance = QtCore.pyqtSignal(ViolanceWarningResult)
    signal_record_video = QtCore.pyqtSignal(str)
    
    def __init__(self, in_buffer: Queue, sequence_buffer: Queue, information_buffer: Queue, camera: Camera):
        super().__init__()
        self._thread_active = False
        self._in_buffer = in_buffer
        self._sequence_buffer = sequence_buffer
        self._output_buffer = Queue()
        self._information_buffer = information_buffer
        
        self.config = Config()
        self._sequence_buffer_size = 1
        self._output_buffer_size = 8
        self._sequence_length = 8
        self._step = 8
        self._window_size = self.config.WINDOW_SIDE
        self._violence_warning_threshold = self.config.VIOLENCE_THRESHOLD
        self._img_size = (224, 224)
        self._classes = ["CO DANH NHAU", ""]
        self.__camera = camera
        self.start_push_event_time = time.time()
        self.record_stream = deque(maxlen=(self.config.RECORD_FPS * self.config.RECORD_TIME))
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

    def start_record(self, video_path):
        self.start_record_time = time.time()
        self.signal_record_video.emit(video_path)
        print("video path: ", video_path)

    def run(self):
        self._thread_active = True
        print("Thread Sequence Running!")

        frames = []
        is_violence = False

        recent_preds = deque(maxlen=self._window_size) 
        violence_confirmed = False
        last_emit_time = 0
        cooldown_seconds = self.config.TIME_TO_PUSH_EVENT + 3  # không emit lại quá sớm

        while self._thread_active:
            if self._in_buffer.qsize() == 0:
                QtCore.QThread.msleep(1)
                continue
            
            if self._information_buffer.qsize() > 0:
                is_violence = self._information_buffer.get()

            frame = self._in_buffer.get()
            
            # Decide whether warning should be emitted
            if self._output_buffer.qsize() < self._output_buffer_size:
                frame_copy = frame.copy()

                # Thêm kết quả mới vào sliding window
                recent_preds.append(is_violence)
                num_violence_frames = sum(recent_preds)
                
                # Kiểm tra có nên emit không
                can_emit = (
                    num_violence_frames >= self._violence_warning_threshold and
                    not violence_confirmed and
                    (time.time() - last_emit_time >= cooldown_seconds)
                )

                if can_emit:
                    violence_confirmed = True
                    last_emit_time = time.time()
                    self.sig_is_violence.emit(True)
                    print("Violence confirmed, emitting warning at ", time.strftime("%H:%M:%S", time.gmtime()))

                    # Draw information
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
                    video_path = os.path.join(save_image_dir, rs.event_id + ".mp4")
                    cv2.imwrite(image_path, frame_copy)

                    media1 = Media()
                    media2 = Media()
                    violance_work_event1 = ViolanceWorkEvent()
                    media1.fileName = f"{rs.event_id}.jpg"
                    media1.filePath = os.path.relpath(image_path,self.config.ROOT_SAVE_DIR)
                    media1.fileType = "1"
                    list_media.append(media1)
                    media2.fileName = f"{rs.event_id}.mp4"
                    media2.filePath = os.path.relpath(video_path,self.config.ROOT_SAVE_DIR)
                    media2.fileType = "2"
                    list_media.append(media2)

                    rs.list_media = list_media
                    rs.violance_work_event = violance_work_event1.serialize()
                    self.signal_violance.emit(rs)
                    self.start_push_event_time = time.time() + self.config.TIME_TO_PUSH_EVENT
                    self.start_record(video_path)
                else:
                    if not is_violence:
                        violence_confirmed = False

                self._output_buffer.put(frame_copy)

            # # Xử lý chuỗi frame để đưa vào sequence
            # if count % self._skip != 1:
            #     continue
            self.record_stream.append(frame)

            # Preprocess frame
            frame = self._preprocess(frame)

            # Put data into sequence
            frames.append(frame)
            if len(frames) == self._sequence_length + 1:
                if self._sequence_buffer.qsize() < self._sequence_buffer_size:
                    self._sequence_buffer.put(frames)
                frames = [frames[-1]]
                                
            QtCore.QThread.msleep(1)

    def stop(self):
        self._thread_active = False
        print("Stop Sequence Thread")
    
    def stop(self):
        self._thread_active = False
        print("Stop Sequence Thread")
    
    def _preprocess(self, frame):
        frame = cv2.resize(frame, self._img_size)
        return frame
