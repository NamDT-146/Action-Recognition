import os
class TrainedModel(object):
    def __init__(self, device="0",
                 model_file=
                     r'/mnt/atin/tttrung/hoa-phat/resources/weights/model_16_m3_0.8888.pth',
                 imgsz=640,
                 conf=0.4,
                 iou_thres=0.45,
                 max_det=20,
                 classes=[0, 1],
                 agnostic_nms=False,
                 half=False):
        
        self.device = device
        self.model_file = model_file
        self.imgsz = imgsz  # inference size (height, width)
        self.conf = conf  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det  # maximum detections per image
        self.classes = classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = agnostic_nms  # class-agnostic NMS
        self.half = half
        self.polygon = []
        
        #addition attribute
        self.zoom = False
        self.time_monitor = "00:00:00"

    @property
    def model_file(self):
        return self._model_file

    @model_file.setter
    def model_file(self, fp):
        self._model_file = fp

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, conf):
        self._conf = conf

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes=[0, 1]):
        self._classes = classes

    @property
    def polygon(self):
        return self._polygon

    @polygon.setter
    def polygon(self, polygon):
        self._polygon = polygon
