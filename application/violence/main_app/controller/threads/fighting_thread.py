from queue import Queue
import time
from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
# from .base_process import BaseProcess
import numpy as np
import torch
import torchvision
from PyQt5.QtCore import QThread
from ..config import Config
from ...model.i3d_flow.utils import load_i3d_model, frames_to_tensor, extract_flow_batch
from ...model.i3d_flow.lite_flownet.run import Network


class FightingThread(QThread):

    def __init__(self, in_buffer: Queue, information_buffer: Queue):
        super().__init__()
        self.max_size_out_buffer = 1
        self._information_buffer = information_buffer
        
        self._in_buffer = in_buffer
        self.config = Config()
        self._device = self.config.DEVICE if torch.cuda.is_available() else "cpu"
        self._conf = float(self.config.CONF)
        self._model = self.config.MODEL_PATH
        self._mode = self.config.MODE
        self._mean = [0.43216, 0.394666, 0.37645]
        self._std = [0.22803, 0.22145, 0.216989]

        # Load I3D model
        print("Loading I3D model...")
        self._model = self._load_model()
    
        # Preload FlowNet model if using flow mode
        self._flownet_model = None
        if self._mode == 'flow':
            print("Loading FlowNet model...")
            self._flownet_model = Network().to(self._device).eval()
            print("FlowNet model loaded successfully")

    def _load_model(self):
        model_ft, _ = load_i3d_model(self._model, self._mode)
        print("Loaded model")
        model_ft.eval()
        
        return model_ft

    # def _load_transform(self):
    #     transform = A.Compose([A.Resize(self._size[1], self._size[0]), A.CenterCrop(
    #         self._center_crop[1],     self._center_crop[0]), A.Normalize(mean=self._mean, std=self._std)])
    #     return transform
    #     # transform = A.Compose([A.Resize(128, 171, always_apply=True),A.CenterCrop(112, 112, always_apply=True),
    #     #             A.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989], always_apply=True)])
    #     # return transform

    def run(self):
        print("Start Fighting Thread")
        self._thread_active = True
        is_violence = False

        prev_processed_frames = None

        while self._thread_active:
            if self._in_buffer.qsize() == 0:
                QtCore.QThread.msleep(1)
                continue
            list_frame = self._in_buffer.get()
            t = time.time()
            if self._mode == 'flow':
                processed_frames = extract_flow_batch(list_frame, device=self._device, flownet_model=self._flownet_model)
                if prev_processed_frames is None:
                    prev_processed_frames = torch.zeros_like(processed_frames)
                composed_processed_frames = torch.cat([prev_processed_frames, processed_frames], dim=0)
                prev_processed_frames = processed_frames
                composed_processed_frames = composed_processed_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
            else:
                frames_tensor = frames_to_tensor(list_frame).to(self._device)
                if prev_processed_frames is None:
                    prev_processed_frames = torch.zeros_like(frames_tensor)
                composed_processed_frames = torch.cat([prev_processed_frames, frames_tensor], dim=0)
                prev_processed_frames = frames_tensor
                composed_processed_frames = composed_processed_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                output = self._model(composed_processed_frames)

            if output.dim() > 2:
                output = output.squeeze(-1)   
            if output.shape[-1] > 1:
                prob_violence = torch.softmax(output, dim=1)[:, 1].item()
                is_violence = prob_violence > self._conf 
            else:
                prob_violence = torch.sigmoid(output).item()
                is_violence = prob_violence > self._conf
            # print(time.time() - t)
            if self._information_buffer.qsize() < 1:
                self._information_buffer.put(is_violence)
                # print(is_violence)
                # print("*" * 20)
