import os
import time

from typing import List
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal

from ...config import ROOT
from ....utils.tools import get_logger
from ....model.camera import Camera
from ....services.camera_api import CameraAPI
# ROOT = get_root_path()
checking_changing_logger = get_logger(os.path.join(ROOT, "resources/log/camera_data_controller.log"))


class CheckHasChange(QThread):
    signal_has_update_camera = pyqtSignal(Camera, str)

    def __init__(self, list_cameras: List[Camera]):
        super().__init__()
        self.__thread_active = False
        self.__original_data: List[Camera] = list_cameras
        self.__updated_data: List[Camera] = []
        # API
        self.__camera_api = CameraAPI()

    def check_has_change(self):
        self.__get_camera_data()
        if self.__original_data and self.__updated_data:
            for cam_data in self.__original_data:
                new_data = self.find_camera_by_id(cam_data.id, self.__updated_data)
                if new_data is not None:
                    is_changed = cam_data.is_different_from(new_data)
                    is_changed_functions = cam_data.is_different_in_functions(new_data)
                    cam_data.merge_data(new_data) #update info
                    
                    if is_changed: # check if any data of camera has change => update info
                        self.write_changing_logger(cam_data, new_data, mode="change")
                        
                    if is_changed_functions:
                        self.signal_has_update_camera.emit(cam_data, "change")  
                else:
                    self.write_changing_logger(cam_data, mode="delete")
                    self.__original_data.remove(cam_data)
                    self.signal_has_update_camera.emit(cam_data, "delete")
            for cam_data in self.__updated_data:
                original_cam_data = self.find_camera_by_id(cam_data.id, self.__original_data)
                # If camera is not in original data, it is new camera => add to list camera data
                if original_cam_data is None:
                    self.write_changing_logger(cam_data, mode="create")
                    self.__original_data.append(cam_data)
                    self.signal_has_update_camera.emit(cam_data, "create")


    def __get_camera_data(self):
        try:
            data = self.__camera_api.get_all_camera_info()
            data = data['data']['data']
            self.__updated_data.clear()
            for e in data:
                camera = Camera.deserialize(e)
                self.__updated_data.append(camera)
                
        except:
            print("Error When Get Camera Data, Retry after 10s")
            checking_changing_logger.error("Error When Get Camera Data")
            checking_changing_logger.error(datetime.now())        

    @staticmethod
    def find_camera_by_id(id_camera, list_camera: List[Camera]) -> Camera:
        if list_camera:
            for e in list_camera:
                if e.id == id_camera:
                    return e
        return None
    
    @staticmethod
    def write_changing_logger(*args: Camera, mode="create"):
        if mode not in ["create", "change", "delete"]:
            raise ValueError("Mode must be create, change or delete")
        
        if mode == "create" or mode == "delete":
            if mode == "delete":
                state = "Old"
            else:
                state = "New"
            camera_data = args[0]
            checking_changing_logger.info("============================")
            checking_changing_logger.info(f"The camera with id {camera_data.id} has {mode}")
            checking_changing_logger.info(f"+++ {state} data: {camera_data.toString()}")
            checking_changing_logger.info("============================")
            
        else:
            old_cam_data, new_cam_data = args
            checking_changing_logger.info("============================")
            checking_changing_logger.info(f"The camera with id {old_cam_data.id} has change")
            checking_changing_logger.info(f"+++ Old data: {old_cam_data.toString()}")
            checking_changing_logger.info(f"The camera with id {old_cam_data.id} has update")
            checking_changing_logger.info(f"+++ New data: {new_cam_data.toString()}")
            checking_changing_logger.info("============================")
    
    def run(self):
        self.__thread_active = True
        while self.__thread_active:
            self.check_has_change()
            time.sleep(10) 
            
    def stop(self):
        self.__thread_active = False
