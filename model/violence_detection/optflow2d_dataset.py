import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

def load_rgb_frames(rgb_path, start=0, num=None):
    frames = np.load(rgb_path)
    
    # Select subset of frames if specified
    if num is not None:
        end = min(start + num, frames.shape[0])
        frames = frames[start:end]
    
    return frames

def load_flow_frames(flow_path, start=0, num=None):
    flow = np.load(flow_path)
    
    # Select subset of frames if specified
    if num is not None:
        end = min(start + num, flow.shape[0])
        flow = flow[start:end]
    
    return flow


def save_denormalized_frames_from_loader(loader, results_dir):
    """
    Save denormalized frames from a DataLoader to results_dir as x_y.jpg,
    where x is the batch/video index and y is the frame index.
    Assumes input is [B, C, T, H, W] and values are in [0, 1] or [-1, 1].
    """
    os.makedirs(results_dir, exist_ok=True)
    for x, (videos, labels) in enumerate(loader):
        # videos: [B, C, T, H, W]
        videos = videos.cpu()
        B, C, T, H, W = videos.shape
        for b in range(B):
            for y in range(T):
                frame = videos[b, :, y, :, :]  # [C, H, W]
                # Denormalize: if in [-1,1], convert to [0,255]; if in [0,1], convert to [0,255]
                np_frame = frame.numpy()
                if np_frame.min() < 0:
                    np_frame = ((np_frame + 1) / 2.0) * 255.0
                else:
                    np_frame = np_frame * 255.0
                np_frame = np_frame.clip(0, 255).astype(np.uint8)
                # Convert from CHW to HWC and BGR for OpenCV
                if C == 1:
                    np_frame = np_frame[0]
                else:
                    np_frame = np.transpose(np_frame, (1, 2, 0))
                    if np_frame.shape[2] == 3:
                        np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(results_dir, f"{x}_{y}.jpg")
                cv2.imwrite(out_path, np_frame)

def save_optflow_frames_as_rgb(loader, results_dir):

    def flow_to_image(flow):
        """
        Converts flow (H, W, 2) into a RGB image (H, W, 3).
        """
        u = flow[..., 0]
        v = flow[..., 1]
        rad = np.sqrt(u ** 2 + v ** 2)
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        img[..., 0] = (np.arctan2(v, u) / (2.0 * np.pi) + 0.5) % 1.0
        img[..., 1] = 1.0
        img[..., 2] = np.clip(rad / (rad_max + epsilon), 0, 1)
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb

    os.makedirs(results_dir, exist_ok=True)
    for x, (videos, labels) in enumerate(loader):
        # videos: [B, C, T, H, W], C==2 for optical flow
        videos = videos.cpu().numpy()
        B, C, T, H, W = videos.shape
        assert C == 2, "Expected optical flow with 2 channels (u,v)"
        for b in range(B):
            for y in range(T):
                flow = np.transpose(videos[b, :, y, :, :], (1, 2, 0))  # (H, W, 2)
                img_rgb = flow_to_image(flow)
                out_path = os.path.join(results_dir, f"{x}_{y}.jpg")
                cv2.imwrite(out_path, img_rgb)

def save_both_rgb_and_flow_frames_as_rgb(loader, results_dir):

    def flow_to_image(flow):
        u = flow[..., 0]
        v = flow[..., 1]
        rad = np.sqrt(u ** 2 + v ** 2)
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        img[..., 0] = (np.arctan2(v, u) / (2.0 * np.pi) + 0.5) % 1.0
        img[..., 1] = 1.0
        img[..., 2] = np.clip(rad / (rad_max + epsilon), 0, 1)
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb

    os.makedirs(results_dir, exist_ok=True)
    for x, ((rgb_videos, flow_videos), labels) in enumerate(loader):
        # rgb_videos: [B, 3, T, H, W], flow_videos: [B, 2, T, H, W]
        rgb_videos = rgb_videos.cpu().numpy()
        flow_videos = flow_videos.cpu().numpy()
        B, C_rgb, T, H, W = rgb_videos.shape
        _, C_flow, _, _, _ = flow_videos.shape
        assert C_rgb == 3 and C_flow == 2, "Expected RGB (3 channels) and Flow (2 channels)"
        for b in range(B):
            for y in range(T):
                # Save RGB frame
                rgb_frame = rgb_videos[b, :, y, :, :]  # [3, H, W]
                np_rgb = np.transpose(rgb_frame, (1, 2, 0))  # HWC
                np_rgb = (np_rgb * 255.0).clip(0, 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
                # Save Flow frame as RGB
                flow = np.transpose(flow_videos[b, :, y, :, :], (1, 2, 0))  # (H, W, 2)
                flow_rgb = flow_to_image(flow)
                # Save with naming convention
                rgb_path = os.path.join(results_dir, f"batch{x}_video{b}_rgb_{y}.jpg")
                flow_path = os.path.join(results_dir, f"batch{x}_video{b}_flow_{y}.jpg")
                both_path = os.path.join(results_dir, f"batch{x}_video{b}_both_{y}.jpg")
                cv2.imwrite(rgb_path, rgb_bgr)
                cv2.imwrite(flow_path, flow_rgb)
                # Optionally, concatenate for visualization
                both = np.concatenate([rgb_bgr, flow_rgb], axis=1)
                cv2.imwrite(both_path, both)

def video_to_tensor(pic):
    """Convert a numpy.ndarray (T x H x W x C) to tensor (C x T x H x W)"""
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class ViolenceDataset(data.Dataset):

    def __init__(self, 
                 split_file, 
                 split='train', 
                 mode='rgb', 
                 num_frames=64, 
                 transforms=None,
                 random_start=True,
                 flow_mag_threshold=0.2,
                 as_flow_rgb=False,
                 val_ratio=0.1,
                 use_val=False):

        self.split_file = split_file
        self.split = split
        self.mode = mode
        self.num_frames = num_frames
        self.transforms = transforms
        self.random_start = random_start
        self.flow_mag_threshold = flow_mag_threshold
        self.as_flow_rgb = as_flow_rgb 
        
        # Load dataset info from CSV
        self.data_info = pd.read_csv(split_file)
        self.data_info = self.data_info[self.data_info['split'] == split]
        
        self.val_ratio = val_ratio
        self.use_val = use_val

        self.data_info = pd.read_csv(split_file)
        self.data_info = self.data_info[self.data_info['split'] == split]

        # Split train into train/val if needed
        if split == 'train' and val_ratio > 0:
            train_idx, val_idx = train_test_split(
                self.data_info.index, 
                test_size=val_ratio, 
                random_state=42, 
                stratify=self.data_info['label']
            )
            if use_val:
                self.data_info = self.data_info.loc[val_idx]
            else:
                self.data_info = self.data_info.loc[train_idx]

            print(f"Loaded {len(self.data_info)} {split}{' (val)' if use_val else ''} samples in {mode} mode")

        
        print(f"Loaded {len(self.data_info)} {split} samples in {mode} mode")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (video_tensor, label)
        """
        # Get info for this sample
        sample = self.data_info.iloc[index]
        rgb_path = sample['rgb_path']
        flow_path = sample['flow_path']
        label = int(sample['label'])
        
        # Determine total frames available
        if self.mode in ['rgb', 'both']:
            all_frames = np.load(rgb_path)
            total_frames = len(all_frames)
        else:
            all_frames = np.load(flow_path)
            total_frames = len(all_frames)
        
        # Determine starting frame
        if self.random_start:
            if total_frames <= self.num_frames:
                start_f = 0
            else:
                start_f = random.randint(0, total_frames - self.num_frames)
        else:
            start_f = 0  # Always start from the beginning for deterministic results
        
        if self.mode == 'rgb':
            frames = load_rgb_frames(rgb_path, start=start_f, num=self.num_frames)
            if self.transforms:
                frames = self.transforms(frames)
            video = video_to_tensor(frames)
            video = video.float() / 255.0

        elif self.mode == 'flow':
            frames = load_flow_frames(flow_path, start=start_f, num=self.num_frames)
       
            frames = self._threshold_flow(frames)
            
            if self.as_flow_rgb:
                # Convert each flow frame to RGB using the same logic as save_optflow_frames_as_rgb
                flow_rgb_frames = []
                for t in range(frames.shape[0]):
                    flow = frames[t]  # (H, W, 2)
                    flow_rgb = self._flow_to_image(flow)  # (H, W, 3), uint8
                    flow_rgb_frames.append(flow_rgb)
                flow_rgb_frames = np.stack(flow_rgb_frames, axis=0)  # (T, H, W, 3)
                if self.transforms:
                    flow_rgb_frames = self.transforms(flow_rgb_frames)
                video = video_to_tensor(flow_rgb_frames)  # (3, T, H, W)
                video = video.float() / 255.0    
            else:
                if self.transforms:
                    frames = self.transforms(frames)
                video = video_to_tensor(frames)
                video = video.float()

        elif self.mode == 'both':
            rgb_frames = load_rgb_frames(rgb_path, start=start_f, num=self.num_frames)
            flow_frames = load_flow_frames(flow_path, start=start_f, num=self.num_frames)
            flow_frames = self._threshold_flow(flow_frames)
            if self.transforms:
                rgb_frames = self.transforms(rgb_frames)
                flow_frames = self.transforms(flow_frames)
            rgb_tensor = video_to_tensor(rgb_frames)
            flow_tensor = video_to_tensor(flow_frames)
            rgb_tensor = rgb_tensor.float() / 255.0
            flow_tensor = flow_tensor.float()
            rgb_tensor = rgb_tensor.permute(1, 0, 2, 3)    # (T, C, H, W)
            flow_tensor = flow_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
            return (rgb_tensor, flow_tensor), torch.tensor(label, dtype=torch.long)

        video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
        return video, torch.tensor(label, dtype=torch.long)


    def __len__(self):
        return len(self.data_info)
    
    def _threshold_flow(self, frames):
        mag = np.sqrt(np.square(frames[..., 0]) + np.square(frames[..., 1]))
        mask = mag >= self.flow_mag_threshold
        frames[~mask] = 0
        return frames
    
    def _flow_to_image(self, flow):

        u = flow[..., 0]
        v = flow[..., 1]
        rad = np.sqrt(u ** 2 + v ** 2)
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        img[..., 0] = (np.arctan2(v, u) / (2.0 * np.pi) + 0.5) % 1.0
        img[..., 1] = 1.0
        img[..., 2] = np.clip(rad / (rad_max + epsilon), 0, 1)
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb

def create_data_loaders(split_file, batch_size=8, num_frames=64, num_workers=4, mode='both', flow_mag_threshold=0.2, as_flow_rgb=False):
    """
    Create data loaders for training and testing
    
    Args:
        split_file (str): Path to CSV file with video info
        batch_size (int): Batch size
        num_frames (int): Number of frames per clip
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader_rgb, train_loader_flow, test_loader_rgb, test_loader_flow)
    """
    # Create datasets
    train_dataset = ViolenceDataset(
        split_file=split_file,
        split='train',
        use_val=False,
        val_ratio=0.2,
        mode=mode,
        num_frames=num_frames,
        flow_mag_threshold=flow_mag_threshold,
        as_flow_rgb=as_flow_rgb
    )
    
    validate_dataset = ViolenceDataset(
        split_file=split_file,
        split='train',
        use_val=True,
        val_ratio=0.2,
        mode=mode,
        num_frames=num_frames,
        flow_mag_threshold=flow_mag_threshold,
        as_flow_rgb=as_flow_rgb
    )
    
    test_dataset = ViolenceDataset(
        split_file=split_file,
        split='test',
        mode=mode,
        num_frames=num_frames,
        flow_mag_threshold=flow_mag_threshold,
        as_flow_rgb=as_flow_rgb
    )
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    validate_loader = data.DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, validate_loader, test_loader


# Example usage
if __name__ == '__main__':
    # Load data
    split_file = 'data/precomputed/split_info.csv'
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        split_file=split_file,
        batch_size=16,
        num_frames=16,
        num_workers=4,
        mode='flow',
        flow_mag_threshold=0.2,
        as_flow_rgb=True
    )
    
    print(f"Train loader size: {len(train_loader)} batches")
    print(f"Validation loader size: {len(val_loader)} batches")
    print(f"Test loader size: {len(test_loader)} batches")  
    
    start = time.time()
    for videos, labels in train_loader:
        print(f"Batch size: {videos.size(0)}, Video shape: {videos.shape}, Labels: {labels}")
        break
    end = time.time()
    print(f"Time taken to load one batch: {end - start:.2f} seconds")