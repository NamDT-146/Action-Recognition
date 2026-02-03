from datetime import datetime
from PyQt5.Qt import QUuid
from typing import List


class Media:
    def __init__(self):
        self.fileName = ""
        self.filePath = ""
        self.fileType = "1"
    
    def serialize(self):
        data = {
            "fileName": self.fileName,
            "filePath": self.filePath,
            "fileType": self.fileType
        }
        return data
    
class ViolanceWorkEvent:
    def __init__(self):
        self.violance_type = "1"
        self.name = "Phat hien danh nhau"
    def serialize(self):
        data = {
            "violanceType": self.violance_type, 
            "ViolanceName": self.name 
        }
        return data


class BaseResult(object):
    def __init__(self):
        self.event_id = QUuid.createUuid().toString(QUuid.StringFormat.WithoutBraces)
        self.access_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.event_type_id =""
        self.warning_level = ""
        self.area_id = ""
        self.comp_id = ""
        self.device_id = ""
        self.list_media: List[Media] = []
        self.violance_work_event: ViolanceWorkEvent = {}
