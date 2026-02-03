import os
import time

from queue import Queue
from PyQt5 import QtCore
from datetime import datetime
from requests.exceptions import ConnectTimeout

from ...config import ROOT
from ....utils.tools import get_logger
from ....services.violance_api import ViolenceWarningAPI
from ....model.violance_result import ViolanceWarningResult
from ....model.camera import Camera
# ROOT = get_root_path()
api_log = get_logger(file_name=os.path.join(ROOT,'resources','log','api_fs.log'))

class EventAPIThread(QtCore.QThread):
    def __init__(self, camera: Camera):
        super().__init__()
        self.__thread_active = False
        self.__violance_api = ViolenceWarningAPI()
        self.__list_event = Queue(maxsize=10)
        self._camera = camera

    def append_to_list_event(self, rs: ViolanceWarningResult):
        self.__list_event.put(rs)

    def post_event(self, rs: ViolanceWarningResult):
        rs_serialize = ViolanceWarningResult.serialize(rs)
        print("Violance Event Serialize", rs_serialize)
        
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        api_log.info("{}--{}".format(dt, rs_serialize))
        
        try:
            response = self.__violance_api.post_violance_event(rs_serialize)
            api_log.info("{}--{}".format(dt, response))
            print('Response: ', response)
        except ConnectTimeout as error:
            api_log.info("{}--{}".format(dt, error))
            print("------------------",error)
            self.append_to_list_event(rs)

        except Exception as error:
            api_log.info("{}--{}".format(dt, error))
            print(error)
            print("Error When Post Violance Event: {}".format(response))

        # # Gửi lên hệ thống Mobifone
        # try:
        #     response_mb = self.__violance_api.post_violance_event_mobifone(rs_serialize)
        #     api_log.info(f"{dt}--Mobifone--{response_mb}")
        #     print('Response (Mobifone): ', response_mb)
        # except ConnectTimeout as error:
        #     api_log.info(f"{dt}--Mobifone Timeout--{error}")
        #     print("Mobifone Timeout:", error)
        # except Exception as error:
        #     api_log.info(f"{dt}--Mobifone Error--{error}")
        #     print("Mobifone Error:", error)


    def run(self):
        print("ThreadAPI: Start")
        self.__thread_active = True
        while self.__thread_active:
            if self.__list_event.empty():
                time.sleep(0.1)
                continue
            rs = self.__list_event.get()
            self.post_event(rs)
            time.sleep(0.01)

    def stop(self):
        self.__thread_active = False
        print("ThreadAPI: Stop")
