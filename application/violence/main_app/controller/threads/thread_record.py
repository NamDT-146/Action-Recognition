import cv2
import time
import ffmpeg
import numpy as np

from PyQt5 import QtCore
from collections import deque

from ...model.camera import Camera
from ..config import Config



class RecordThread(QtCore.QThread):
    def __init__(self, record_stream: deque, camera: Camera):
        super().__init__()
        self.__thread_active = False
        self.__w = 1280
        self.__h = 720
        self.__record_stream = record_stream
        self.__camera = camera
        self.__video_path = ""
        self.config = Config()
    
    @property
    def record_stream(self):
        return self.__record_stream
    
    @property
    def video_path(self):
        return self.__video_path

    @property
    def camera_config(self):
        return self.__camera

    @camera_config.setter
    def camera_config(self, camera: Camera):
        self.__camera = camera
    
    @property
    def width(self):
        return self.__w

    @width.setter
    def width(self, w):
        self.__w = w

    @property
    def height(self):
        return self.__h

    @height.setter
    def height(self, h):
        self.__h = h

    # @property
    # def __thread_active(self):
    #     return self.__thread_active

    def _init_arg(self):
        self._process = (
            ffmpeg
            .input('pipe:0', framerate='{}'.format(self.config.RECORD_FPS), format='rawvideo', pix_fmt='bgr24',
                   s='{}x{}'.format(int(self.__w), int(self.__h)),loglevel='quiet')
            .output(f'{self.__video_path}', vcodec='h264', pix_fmt='nv21')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __record(self, array_frame):
        self._init_arg()

        try:
            for frame in array_frame:
                frame = cv2.resize(frame, (self.__w, self.__h))
                self._process.stdin.write(frame.astype(np.uint8).tobytes())
            self._process.communicate(str.encode("q"))

        except Exception as e:
            print(e)

    def append_video_path(self, video_path: str):
        self.__video_path = video_path

    def run(self):
        self.__thread_active = True
        print("Thread Record: Start")
        while self.__thread_active:
            if not self.__video_path:   # check video path is empty string
                time.sleep(0.02)
                continue

            print("Recording and Saving Video with Path: ", self.__video_path)
            list_frame = list(self.__record_stream)
            self.__record(list_frame)
            self.__video_path = ""
            time.sleep(0.001)

    def stop(self):
        self.__thread_active = False
        print("Thread Record: Stop")
