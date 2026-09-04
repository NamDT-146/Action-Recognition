from typing import List
from PyQt5.QtWidgets import QMainWindow

from ..model.camera import Camera
from ..services.camera_api import CameraAPI

from .camera.c_violence_wg_camera import WgViolence
from .threads.api.check_has_changed_camera import CheckHasChange


class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cameraAPI = CameraAPI()
        self.list_camera:List[Camera] = []
        self.list_camera_objects:List[WgViolence] = []
        self.event_type_id = '202'
        self._init_cameras()
        self.start_cameras()
        self.check_has_changed_camera = CheckHasChange(
            list_cameras=self.list_camera)
        self.check_has_changed_camera.start()
        
        self.check_has_changed_camera.signal_has_update_camera.connect(self.update_camera_object)

    def _init_cameras(self):
        self.list_camera = self._init_config_camera()
        if self.list_camera:
            for camera in self.list_camera:
                if int(camera.event_type_id) == int(self.event_type_id):   
                    new_camera = WgViolence(camera)
                    self.list_camera_objects.append(new_camera)

    def _init_config_camera(self):
        configs: List[Camera] = []
        data_json = self.cameraAPI.get_all_camera_info()
        print("####### data_json: ", data_json)
        data_json = data_json['data']['data']
        print("####### data_json: ", data_json)
        for e in data_json:
            new_cf = Camera.deserialize(e)
            configs.append(new_cf)
        return configs

    def start_cameras(self):
        print('Number of cameras:', len(self.list_camera_objects))
        if len(self.list_camera_objects):
            for camera_obj in self.list_camera_objects:
                print("start camera: ", camera_obj.camera.id)
                camera_obj.start()
                # Put camera status to 1 (running)
                print("########## camera_obj.camera.id: ", camera_obj.camera.id)
                self.cameraAPI.put_camera_status(camera_obj.camera.id, 1)
                
    def update_camera_object(self, camera: Camera, mode=""):
        # Find existing camera object
        old_cam_obj = None
        for cam_obj in self.list_camera_objects:
            if cam_obj.camera.id == camera.id:
                old_cam_obj = cam_obj
                break
        if mode == "delete":
            if old_cam_obj is not None:
                print(f"Removing camera with ID: {camera.id}")
                old_cam_obj.stop()
                self.list_camera_objects.remove(old_cam_obj)
                self.cameraAPI.put_camera_status(camera.id, 0)  # Set status to offline
            return
        elif mode == "change":
            if old_cam_obj is not None:
                # Check if this camera still has fire smoke detection function
                if int(self.event_type_id) == int(camera.event_type_id):
                    print(f"Updating camera with ID: {camera.id}")
                    # Stop old camera object
                    old_cam_obj.stop()
                    # Update camera data
                    old_cam_obj.camera.merge_data(camera)
                    # Restart with new configuration
                    old_cam_obj.start()
                    self.cameraAPI.put_camera_status(camera.id, 1)
                else:
                    # Remove camera if it no longer has fire smoke detection function
                    print(f"Removing Fire Smoke Detect Camera with ID: {camera.id}")
                    old_cam_obj.stop()
                    self.list_camera_objects.remove(old_cam_obj)
                    self.cameraAPI.put_camera_status(camera.id, 0)
            return
        elif mode == "create":
            # Add new camera if it has fire smoke detection function
            if int(self.event_type_id) == int(camera.event_type_id):
                print(f"Adding new camera with ID: {camera.id}")
                new_cam_obj = Camera(camera)
                self.list_camera_objects.append(new_cam_obj)
                new_cam_obj.start()
            self.cameraAPI.put_camera_status(camera.id, 1)