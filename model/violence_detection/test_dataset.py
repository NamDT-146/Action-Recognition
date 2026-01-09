import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import random

def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

def load_rgb_frames(rgb_path, start=0, num=None):
    frames = np.load(rgb_path)
    
    if num is not None:
        end = min(start + num, frames.shape[0])
        frames = frames[start:end]
    
    return frames

def load_flow_frames(flow_path, start=0, num=None):
    flow = np.load(flow_path)
    
    if num is not None:
        end = min(start + num, flow.shape[0])
        flow = flow[start:end]
    
    return flow

class ViolenceDataset(data.Dataset):
    """
    Dataset for violence detection using precomputed RGB and optical flow
    """
    def __init__(self, 
                 split_file, 
                 split='train', 
                 mode='rgb', 
                 num_frames=64, 
                 transforms=None,
                 random_start=True):
        """
        Args:
            split_file (str): Path to CSV file with video info
            split (str): 'train' or 'test'
            mode (str): 'rgb', 'flow', or 'both'
            num_frames (int): Number of frames to load per clip
            transforms (callable): Optional transform for frames
            random_start (bool): Whether to use random or fixed starting frame
        """
        self.split_file = split_file
        self.split = split
        self.mode = mode
        self.num_frames = num_frames
        self.transforms = transforms
        self.random_start = random_start
        
        # Load dataset info from CSV
        self.data_info = pd.read_csv(split_file)
        self.data_info = self.data_info[self.data_info['split'] == split]
        
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
        
        # Load appropriate frames based on mode
        if self.mode == 'rgb':
            frames = load_rgb_frames(rgb_path, start=start_f, num=self.num_frames)
            # Apply transforms if available
            if self.transforms:
                frames = self.transforms(frames)
            # Convert to tensor [C, T, H, W]
            video = video_to_tensor(frames)
            video = video.float() / 255.0

            
        elif self.mode == 'flow':
            frames = load_flow_frames(flow_path, start=start_f, num=self.num_frames)
            # Apply transforms if available
            if self.transforms:
                frames = self.transforms(frames)
            # Convert to tensor [C, T, H, W]
            video = video_to_tensor(frames)
            video = video.float()  # Add this line to create a fresh tensor

            
        elif self.mode == 'both':
            # Load both RGB and flow
            rgb_frames = load_rgb_frames(rgb_path, start=start_f, num=self.num_frames)
            flow_frames = load_flow_frames(flow_path, start=start_f, num=self.num_frames)
            
            # Apply transforms if available
            if self.transforms:
                rgb_frames = self.transforms(rgb_frames)
                flow_frames = self.transforms(flow_frames)
                
            # Convert to tensors
            rgb_tensor = video_to_tensor(rgb_frames)
            flow_tensor = video_to_tensor(flow_frames)
            
            rgb_tensor = rgb_tensor.float() / 255.0
            flow_tensor = flow_tensor.float()
            # Return tuple of tensors
            return (rgb_tensor, flow_tensor), torch.tensor(label, dtype=torch.long)
        
        # For single stream, return single tensor
        return video, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data_info)


def create_data_loaders(split_file, batch_size=8, num_frames=64, num_workers=4):
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
    train_dataset_rgb = ViolenceDataset(
        split_file=split_file,
        split='train',
        mode='rgb',
        num_frames=num_frames
    )
    
    train_dataset_flow = ViolenceDataset(
        split_file=split_file,
        split='train',
        mode='flow',
        num_frames=num_frames
    )
    
    test_dataset_rgb = ViolenceDataset(
        split_file=split_file,
        split='test',
        mode='rgb',
        num_frames=num_frames,
        random_start=False  # Deterministic for testing
    )
    
    test_dataset_flow = ViolenceDataset(
        split_file=split_file,
        split='test',
        mode='flow',
        num_frames=num_frames,
        random_start=False  # Deterministic for testing
    )
    
    # Create data loaders
    train_loader_rgb = data.DataLoader(
        train_dataset_rgb,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    train_loader_flow = data.DataLoader(
        train_dataset_flow,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_rgb = data.DataLoader(
        test_dataset_rgb,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_flow = data.DataLoader(
        test_dataset_flow,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader_rgb, train_loader_flow, test_loader_rgb, test_loader_flow


# Example usage
if __name__ == '__main__':
    # Load data
    split_file = 'data/precomputed/split_info.csv'
    
    # Create data loaders
    train_loader_rgb, train_loader_flow, test_loader_rgb, test_loader_flow = create_data_loaders(
        split_file=split_file,
        batch_size=8,
        num_frames=64,
        num_workers=4
    )
    
    # Check shapes
    for videos, labels in train_loader_rgb:
        print(f"RGB Batch shape: {videos.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break
        
    for videos, labels in train_loader_flow:
        print(f"Flow Batch shape: {videos.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    
    
