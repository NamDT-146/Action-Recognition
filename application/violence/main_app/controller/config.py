import os
import yaml
import sys
def get_root_path():
    """Trả về đường dẫn gốc (cùng thư mục với .py hoặc .exe)"""
    if getattr(sys, 'frozen', False):
        # sys.executable sẽ trỏ đến file .exe.
        return os.path.dirname(sys.executable)
    else:
        # Khi chạy từ source code, __file__ là đường dẫn đến file config.py này.
        return os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
ROOT = get_root_path()
config_file_path = os.path.join(ROOT, "resources/config/config.yaml")

class Config():
    def __init__(self):
        self.MODEL_PATH = ""
        self.DEVICE = "cuda:0"
        self.HALF = False
        
        self.USERNAME = ""
        self.PASSWORD = ""

        self.WEB_HOST = ""
        self.SERVER_NAME = ""
        self.MEDIA_SERVER = ""
        self.SOCKET_SERVER = ""
        self.CONF = ""
        self.TIME_TO_PUSH_EVENT = 2
        self.STREAM_SIZE = [960, 540]

        self.ROOT_SAVE_DIR = ""
        self.ROOT_SAVE_IMAGE_DIR = ""

        self.read_file_config()

    def read_file_config(self):
        with open(config_file_path, 'r',encoding="utf8") as f:
            try:
                self.data = yaml.safe_load(f)
                # Do something with the data
            except yaml.YAMLError as exc:
                print(exc)
        for key, value in self.data.items():
            self.__setattr__(key, value)
        self.MODEL_PATH = os.path.join(ROOT, self.MODEL_PATH)