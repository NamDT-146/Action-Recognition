import time
import requests

from ..controller.config import Config

EXPRISE_TIME = 900 # 15 minutes


class BaseAPI(object):
    def __init__(self):
        super().__init__()
        self._token = None
        self.config = Config()
        self.compId = 39
        self._start_time = 0

    def get_access_token_from_web(self):
        api_get_token = self.config.WEB_HOST + "/Service/api/token/auth"
        print(api_get_token)
        try:
            dict_data = {
                "client_id": "IAC_Cloud",
                "client_secret": "1a82f1d60ba6353bb64a8fb4b05e4bc4",
                "grant_type": "password",
                "username": self.config.USERNAME,
                "password": self.config.PASSWORD
            }
            rs = requests.post(api_get_token, json=dict_data, timeout=10)
            rs = rs.json()
            self._token = rs['access_token']
            self._start_time = time.time()
            return True
        except Exception as e:
            print("Error When Get Access Token: ", e)
            return False
        
    
    def get_access_token(self):
        if self._token is not None:
            if time.time() - self._start_time > EXPRISE_TIME:
                self.get_access_token_from_web()
            else:
                return True
        else:
            self.get_access_token_from_web()
