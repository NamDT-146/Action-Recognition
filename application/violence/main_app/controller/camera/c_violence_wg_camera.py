from PyQt5.QtWidgets import QWidget
from ...model.camera import Camera
from PyQt5 import QtGui, QtWidgets, QtCore
from queue import Queue

from ..threads.thread_capture import CaptureThread
from ..threads.fighting_thread import FightingThread
from ..threads.sequence_thread import SequenceThread
from ..threads.thread_stream import StreamThread
from ..threads.thread_record import RecordThread
from ..threads.api.thread_violance_event_api import EventAPIThread
from ...services.socket_client import ConnectToWSS
class WgViolence(QWidget):
    def __init__(self, camera: Camera):
        super().__init__()

        self.camera: Camera = camera
        self.list_threads = []
        self._init_thread(self.camera)
        self.connect_signal()
        self.start()
        
    def connect_signal(self):
        self._sequence_thread.signal_violance.connect(self.api_thread.append_to_list_event)
        # self.detect_thread.signal_record_video.connect(self.record_thread.append_video_path)
        self._sequence_thread.signal_record_video.connect(self.record_thread.append_video_path)

    def _init_thread(self, camera):
        self._sequence_buffer = Queue()
        self._information_buffer = Queue()
        self._output_buffer = Queue()

        self.thread_capture = CaptureThread(camera)
        self._sequence_thread = SequenceThread(
             self.thread_capture.capture_queue, self._sequence_buffer, self._information_buffer, self.camera)
        self._process_thread = FightingThread(
            self._sequence_buffer, self._information_buffer)
        self.stream_thread = StreamThread(self._sequence_thread.out_buffer, self.camera)
        self.socket_client = ConnectToWSS(self.camera)
        self.api_thread = EventAPIThread(self.camera)
        self.record_thread = RecordThread(self._sequence_thread.record_stream, self.camera)
        self.list_threads = [self.thread_capture, self._sequence_thread,
                             self._process_thread, self.stream_thread, self.api_thread, self.record_thread
                            ]
        
   
    def start(self):
        for thread in self.list_threads:
            thread.start()

    def stop(self):
        for thread in self.list_threads:
            thread.stop()
