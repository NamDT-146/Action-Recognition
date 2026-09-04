import json
import requests

from .base_api import BaseAPI


class CameraAPI(BaseAPI):
    def __init__(self):
        super().__init__()
        self._url_param = f"page=1&itemsPerPage=999&sortBy=id&sortDesc=true&compId={self.compId}&serverName={self.config.SERVER_NAME}&eventTypeId=202"
        # self._url_param = f"page=1&itemsPerPage=999&sortBy=id&sortDesc=true&compId={self.compId}&eventTypeId=202"
        # self._url_param = f"page=1&itemsPerPage=10&sortBy=&codeName=&compId={self.compId}&areaId=&status=1&eventTypeId=202"

    def get_all_camera_info(self):
        get_all_camera_info_api_url = f"{self.config.WEB_HOST}/Service/api/Device?{self._url_param}"
        # get_all_camera_info_api_url = "http://192.168.1.144:42048/Service/api/device?page=1&itemsPerPage=10&sortBy=&codeName=&compId=39&areaId=&status=1&eventTypeId=202"
        self.get_access_token()
        try:
            data = requests.get(get_all_camera_info_api_url,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {self._token}"})
            # print("All camera data: ", data.json())
            data_str = json.dumps(data.json(), indent=4, ensure_ascii=False)
            return json.loads(data_str)
        except Exception as e:
            print("Error When Get All Camera Info: ", e)
            return {}

    def get_accesstime_by_id(self, id):
        get_accesstime_by_id_api_url = f"{self.config.WEB_HOST}/Service/api/accesstimeseg/{id}"
        self.get_access_token()
        try:
            data = requests.get(get_accesstime_by_id_api_url,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {self._token}"})
            data_str = json.dumps(data.json(), indent=4, ensure_ascii=False)
            return json.loads(data_str)
        except Exception as e:
            print("Error When Get Access Time by ID: ", e)
            return {}

    def put_camera_status(self, camera_id, status=1):
        # status = 1: Online, 0: Offline
        
        put_camera_status_api_url = f"{self.config.WEB_HOST}/Service/api/device/status/{camera_id}/{status}"
        self.get_access_token()
        try:
            resp = requests.put(put_camera_status_api_url,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {self._token}"})
            return resp.json()
        except Exception as er:
            # print("Error When Put Camera Status: ", er)
            return {}

