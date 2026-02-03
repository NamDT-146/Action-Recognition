from .base_result import BaseResult
from datetime import datetime


class ViolanceWarningResult(BaseResult):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def serialize(rs:"ViolanceWarningResult"):
        violance_work_event = rs.violance_work_event
        list_media = [media.serialize() for media in rs.list_media]
        data = {
            "eventId": rs.event_id,
            "accessTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "eventTypeId": str(rs.event_type_id),
            "areaId": str(rs.area_id),
            "compId": str(rs.comp_id),
            "deviceId": str(rs.device_id),
            "listMedia": list_media,
            "violanceEvent": violance_work_event
        }
        return data
