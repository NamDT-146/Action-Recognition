import cv2
import time

from queue import Queue
from PyQt5 import QtCore

from ...model.camera import Camera
from ...services.camera_api import CameraAPI


class CaptureThread(QtCore.QThread):
    def __init__(self, camera: Camera):
        super().__init__()
        self.__thread_active = True
        self.capture_queue = Queue(maxsize=10)
        self.__camera = camera
        self._previous_link = self.__camera.link
        self._ms_sleep = 10
        self.camera_api = CameraAPI()

    def setup_cap(self):
        self._cap = cv2.VideoCapture(self.__camera.link)
        self._previous_link = self.__camera.link
        if self._previous_link.endswith((".mp4", ".avi", ".mov")):
            self._ms_sleep = 30

    def run(self):
        print("ThreadCapture: Start")
        self.__thread_active = True
        self.setup_cap()

        while self.__thread_active:
            ret, frame = self._cap.read()
            if not ret:
                self.setup_cap()
                # Set camera status to offline
                if self.__camera.status == 1:
                    self.camera_api.put_camera_status(self.__camera.id, 2)
                    self.__camera.status = 2
                time.sleep(2)
                continue

            if self._previous_link != self.__camera.link:
                self.setup_cap()
                # print("Previous Link: ", self._previous_link)
                print("New Camera Link: ", self.__camera.link)
                continue
            
            # Set camera status to online if it is offline
            if self.__camera.status == 2:
                self.camera_api.put_camera_status(self.__camera.id, 1)
                self.__camera.status = 1

            if not self.capture_queue.full():
                self.capture_queue.put(frame)

            self.msleep(self._ms_sleep)

    def stop(self):
        self.__thread_active = False
        print("ThreadCapture: Stop")
