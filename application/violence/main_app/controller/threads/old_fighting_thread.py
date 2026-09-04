from queue import Queue
import time
from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
from ...model.trained_model import TrainedModel
# from .base_process import BaseProcess
import numpy as np
import torch
import torchvision
import albumentations as A
from PyQt5.QtCore import QThread
from torchvision.models.video import mc3_18, MC3_18_Weights
from ..config import Config
class FightingThread(QThread):

    def __init__(self, in_buffer: Queue, information_buffer: Queue):
        super().__init__()
        self.max_size_out_buffer = 1
        self._information_buffer = information_buffer
        
        self._in_buffer = in_buffer
        # self._model_path = "resources/Weight/suicide100_8_20_v1s.pth"   #hoai
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._skip_frame = 2
        self.config = Config()
        self.conf = self.config.CONF
        self._trained_model = TrainedModel(conf=self.conf)
        self._model = self.config.MODEL_PATH
        self._size = (128, 171)
        self._center_crop = (112, 112)
        self._mean = [0.43216, 0.394666, 0.37645]
        self._std = [0.22803, 0.22145, 0.216989]
        self._transform = self._load_transform()

    @property
    def trained_model(self):
        return self._trained_model

    @trained_model.setter
    def trained_model(self, trained_model: TrainedModel):
        trained_model.conf = self.conf
        self._trained_model = trained_model

    def _load_model(self):
        weights = MC3_18_Weights.DEFAULT  # hoặc MC3_18_Weights.KINETICS400_V1
        model_ft = torchvision.models.video.mc3_18(
            weights=weights, progress=False)  # quang
        # model_ft = torchvision.models.video.r2plus1d_18(weights=True, progress=False)      #hoai
        num_ftrs = model_ft.fc.in_features  # in_features
        # nn.Linear(in_features, out_features)
        model_ft.fc = torch.nn.Linear(num_ftrs, 2)
        model_ft.load_state_dict(torch.load(
            self._model, map_location=torch.device(self._device),weights_only=False))
        model_ft.to(self._device)
        model_ft.eval()
        print("Loaded model")
        
        return model_ft

    def _load_transform(self):
        transform = A.Compose([A.Resize(self._size[1], self._size[0]), A.CenterCrop(
            self._center_crop[1],     self._center_crop[0]), A.Normalize(mean=self._mean, std=self._std)])
        return transform
        # transform = A.Compose([A.Resize(128, 171, always_apply=True),A.CenterCrop(112, 112, always_apply=True),
        #             A.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989], always_apply=True)])
        # return transform

    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self._transform(image=frame)['image']
        return frame

    def _predict(self, frames):
        with torch.no_grad():
            input_frames = np.array(frames)

            # add an extra dimension
            input_frames = np.expand_dims(input_frames, axis=0)

            # transpose to get [1, 3, num_clips, height, width]
            input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

            # convert the frames to tensor
            input_frames = torch.tensor(input_frames, dtype=torch.float32)
            input_frames = input_frames.to(self._device)

            # forward pass to get the predictions
            outputs = self._model(input_frames)

            # get the prediction index
            soft_max = torch.nn.Softmax(dim=1)
            probs = soft_max(outputs.data)
            prob, indices = torch.topk(probs, k=1)

        class_index = indices[0][0].item()
        return class_index

    def detect(self, frames):
        class_index = self._predict(frames)
        return class_index

    def run(self):
        print("Start Fighting Thread")
        self._thread_active = True
        self._model = self._load_model()
        # print(self.conf)
        # print(self._trained_model.conf)
        is_violence = False
        while self._thread_active:
            if self._in_buffer.qsize() == 0:
                QtCore.QThread.msleep(1)
                continue
            list_frame = self._in_buffer.get()
            t = time.time()
            frames = list(map(self._preprocess, list_frame))
            is_violence = self.detect(frames) == 0
            # print(time.time() - t)
            frames = []
            if self._information_buffer.qsize() < 1:
                self._information_buffer.put(is_violence)
                # print(is_violence)
                # print("*" * 20)
