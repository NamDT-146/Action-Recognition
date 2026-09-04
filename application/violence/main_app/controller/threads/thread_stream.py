import cv2
import subprocess
import time

from queue import Queue
from PyQt5 import QtCore

from ...model.camera import Camera
from ..config import Config


class StreamThread(QtCore.QThread):
    def __init__(self, stream_queue: Queue, camera: Camera):
        super().__init__()
        self.__thread_active = False
        self.config = Config()
        self._stream_size = self.config.STREAM_SIZE
        self._camera_url = ""
        self._index = 0
        self.stream_queue = stream_queue
        self.camera = camera
        self.fps = self.config.INPUT_FPS
        self.__previous_rtmp = ""
        self.__ffmpeg_process: subprocess.Popen
        self.link_rtmp = ""
        self.__pre_flag = False

    def __set_args(self, rtmp_link):
        w, h = self._stream_size[0], self._stream_size[1]
        return (
            f"ffmpeg -r {self.fps} -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s {w}x{h} -i pipe:0 "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -tune zerolatency -b:v 4096k "
            f"-f flv {rtmp_link} -loglevel quiet"
        ).split()

    # def __set_args(self, rtmp_link):
    #     w, h = self._stream_size[0], self._stream_size[1]
    #     stream_fps = self.config.INPUT_FPS
    #     return (
    #         f"ffmpeg -r {stream_fps} -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s {w}x{h} -i pipe:0 "
    #         f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -tune zerolatency -b:v 800k "
    #         f"-g {stream_fps} -keyint_min {stream_fps} -sc_threshold 0 -bufsize 800k -maxrate 800k "
    #         f"-flush_packets 1 -max_delay 100000 -fflags nobuffer "
    #         f"-f flv {rtmp_link} -loglevel quiet"
    #     ).split()

    def __create_process(self, rtmp):
        args = self.__set_args(rtmp)
        return subprocess.Popen(args, stdin=subprocess.PIPE)

    def __create_uri_encode(self):
        return f"POC_{self.camera.comp_id}/{self.camera.code}.flv"

    def __create_rtmp_link(self):
        uri_encode = self.__create_uri_encode()
        return f"{self.config.MEDIA_SERVER}/{uri_encode}" 

    def run(self):
        print("ThreadStream: Start")
        self.__thread_active = True
        rtmp_link = self.__create_rtmp_link()
        print("Starting rtmp stream with rtmp: ", rtmp_link)
        self.__ffmpeg_process = self.__create_process(rtmp_link)
        self.__previous_rtmp = rtmp_link
        self.link_rtmp = rtmp_link
        # Check when first frame be writen

        while self.__thread_active:
            if self.stream_queue.empty():
                self.msleep(1)
                continue
            frame = self.stream_queue.get()

            # if not self.camera.is_streaming:
            #     if not self.__pre_flag:
            #         self.__ffmpeg_process.stdin.close()
            #         self.__ffmpeg_process.wait()
            #         self.__pre_flag = True
            #     self.msleep(1000)
            #     continue
            try:
                frame = cv2.resize(frame, (self._stream_size[0], self._stream_size[1]))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                new_rtmp = self.__create_rtmp_link()
                if new_rtmp != self.__previous_rtmp:
                    self.__ffmpeg_process.stdin.close()
                    self.__ffmpeg_process.wait()
                    self.__ffmpeg_process = self.__create_process(new_rtmp)
                    self.__previous_rtmp = new_rtmp
                    self.link_rtmp = new_rtmp
                    continue
                self.__ffmpeg_process.stdin.write(frame.tobytes())
                

            except Exception as e:
                print(e)
                self.__pre_flag = False
                self.__ffmpeg_process.stdin.close()
                self.__ffmpeg_process.wait()
                self.__ffmpeg_process = self.__create_process(rtmp_link)

            self.msleep(1)

        self.__ffmpeg_process.stdin.close()
        self.__ffmpeg_process.wait()

    def stop(self):
        self.__thread_active = False
        print("ThreadStream: Stop")
