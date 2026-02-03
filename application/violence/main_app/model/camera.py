class Camera():
    def __init__(self):
        self.id = None
        self.code = ""
        self.name = ""
        self.link = ""
        self.license = ""
        self.comp_id = None
        self.area_id = None
        self.serverId = None
        self.event_type_id = None
        self.status = None
        self.longitude = None
        self.latitude = None
        self.area_name = ""
        self.event_type_name = ""
        self.status_name = ""
        self.server_name = ""
    def merge_data(self, __o:"Camera"):
        self.id = __o.id
        self.code = __o.code
        self.name = __o.name
        self.link = __o.link
        self.license = __o.license
        self.comp_id = __o.comp_id
        self.area_id = __o.area_id
        self.serverId = __o.serverId
        self.event_type_id = __o.event_type_id
        self.status = __o.status
        self.longitude = __o.longitude
        self.latitude = __o.latitude
        self.area_name = __o.area_name
        self.event_type_name = __o.event_type_name
        self.status_name = __o.status_name
        self.server_name = __o.server_name

    @staticmethod
    def deserialize(data):
        try:
            new_camera = Camera()
            new_camera.id = data["id"]
            new_camera.code = data["code"]
            new_camera.name = data["name"]
            new_camera.link = data["link"]
            new_camera.license = data["license"]
            new_camera.comp_id = data["compId"]
            new_camera.area_id = data["areaId"]
            new_camera.serverId = data["serverId"]
            new_camera.event_type_id = data["eventTypeId"]
            new_camera.status = data["status"]
            new_camera.longitude = data["longitude"]
            new_camera.latitude = data["latitude"]
            new_camera.area_name = data["areaName"]
            new_camera.event_type_name = data["eventTypeName"]
            new_camera.status_name = data["statusName"]
            new_camera.server_name = data["serverName"]
                  
        except Exception as e:
            print("Error deserialize: ", e)
        return new_camera
    
    def is_different_from(self, __o: 'Camera'):
        is_changed_ = (self.comp_id != __o.comp_id) \
            or (self.code != __o.code) \
            or (self.link != __o.link) \
            or (self.event_type_id != __o.event_type_id)
        # or self.polygon == __o.polygon
        return is_changed_
    
    def is_different_in_functions(self, __o: 'Camera'):
        return self.event_type_id != __o.event_type_id
    
    def toString(self):
        return f"{self.id} {self.link} {self.code} {self.area_name}"
