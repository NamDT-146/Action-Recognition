import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from flownet.run import estimate  # Import from liteflownet
import pandas as pd

class DenseOpticalFlowDataset(Dataset):
    def __init__(self, data_dir, dataset_name, frame_size=512, num_frames=150, one_hot=False):
        """
        Dataset for Violence Detection with Dense Optical Flow
        
        Args:
            data_dir (str): Root directory containing datasets
            dataset_name (str): Name of dataset (will be converted to lowercase)
            frame_size (int): Size to resize frames to (frame_size x frame_size)
            num_frames (int): Number of frames to extract (default: 150)
            one_hot (bool): If True, return one-hot encoded labels instead of binary labels
        """
        self.one_hot = one_hot
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # Define dataset path
        self.dataset_path = os.path.join(data_dir, self.dataset_name)
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        
        self._load_dataset()
        print(f"Loaded {len(self.video_paths)} videos from {self.dataset_name}")
        print(f"Nonviolence: {self.labels.count(0)}, Violence: {self.labels.count(1)}")
    
    def _load_dataset(self):
        """Load video paths and labels from nonviolence and violence folders"""
        
        # Load nonviolence videos (label = 0)
        nonviolence_path = os.path.join(self.dataset_path, "nonviolence")
        if os.path.exists(nonviolence_path):
            for video_file in os.listdir(nonviolence_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                    video_path = os.path.join(nonviolence_path, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(0)
        
        # Load violence videos (label = 1)
        violence_path = os.path.join(self.dataset_path, "violence")
        if os.path.exists(violence_path):
            for video_file in os.listdir(violence_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                    video_path = os.path.join(violence_path, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence with optical flow
        
        Returns:
            sequence (tensor): Shape (3, 150, 512, 512)
                Channel 0: Grayscale of original frame
                Channels 1-2: Optical flow (x, y components)
            label (tensor): Binary label (0 or 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames and compute optical flow
        frames, optical_flows = self._extract_frames_and_flow(video_path)
        
        # Build the sequence tensor: [3, 150, 512, 512]
        sequence = self._build_sequence_tensor(frames, optical_flows)
        
        # Return label as either binary or one-hot encoded
        if self.one_hot:
            # One-hot encoding for 2 classes (0: non-violence, 1: violence)
            one_hot_label = torch.zeros(2, dtype=torch.float)
            one_hot_label[label] = 1.0
            return sequence, one_hot_label
        else:
            return sequence, torch.tensor(label, dtype=torch.long)
    
    def _extract_frames_and_flow(self, video_path):
        """
        Extract frames and compute optical flow
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            tuple: (frames, optical_flows)
                frames: List of grayscale frames [150 frames]
                optical_flows: List of optical flow tensors [149 pairs]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle case where video has no frames or invalid frame count
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has {total_frames} frames")
            cap.release()
            # Return dummy black frames
            dummy_frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
            dummy_flow = np.zeros((self.frame_size, self.frame_size, 2), dtype=np.float32)
            return [dummy_frame] * self.num_frames, [dummy_flow] * (self.num_frames - 1)
        
        # Read all frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to frame_size x frame_size
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_frames.append(gray)
        
        cap.release()
        
        # If we don't have enough frames, use mirroring to extend
        if len(all_frames) < self.num_frames:
            all_frames = self._extend_frames_by_mirroring(all_frames, self.num_frames)
        
        # If we have more frames than needed, select a random subsequence
        if len(all_frames) > self.num_frames:
            start_idx = random.randint(0, len(all_frames) - self.num_frames)
            all_frames = all_frames[start_idx:start_idx + self.num_frames]
        
        # Compute optical flow between consecutive frames
        optical_flows = []
        
        for i in range(len(all_frames) - 1):
            prev_frame = all_frames[i]
            curr_frame = all_frames[i + 1]
            
            # Convert to RGB float tensors (LiteFlowNet expects RGB)
            prev_tensor = self._frame_to_tensor(prev_frame)
            curr_tensor = self._frame_to_tensor(curr_frame)
            
            # Compute optical flow
            with torch.no_grad():
                flow_tensor = estimate(prev_tensor, curr_tensor)
            
            # Convert to numpy array
            flow = flow_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # Normalize flow values to [0, 1] range
            flow_max = 20.0  # Define maximum expected flow value
            flow = np.clip(flow, -flow_max, flow_max)
            flow = (flow + flow_max) / (2 * flow_max)  # Map from [-max_flow, max_flow] to [0, 1]
            
            
            optical_flows.append(flow)
        
        return all_frames, optical_flows
    
    def _frame_to_tensor(self, gray_frame):
        """Convert grayscale frame to RGB tensor for LiteFlowNet"""
        # Repeat grayscale to create RGB
        rgb = np.stack([gray_frame] * 3, axis=-1)
        
        # Convert to float tensor [0,1]
        tensor = torch.FloatTensor(
            np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        )
        
        return tensor
    
    def _extend_frames_by_mirroring(self, frames, target_length):
        """
        Extend frame list by mirroring until we reach target_length
        Example: [1,2,3] -> [1,2,3,3,2,1,1,2,3,...]
        """
        extended = frames.copy()
        reversed_frames = frames[::-1]
        
        while len(extended) < target_length:
            if len(extended) + len(reversed_frames) <= target_length:
                extended.extend(reversed_frames)
                if len(extended) < target_length:
                    extended.extend(frames)
            else:
                # Add only as many frames as needed
                remaining = target_length - len(extended)
                extended.extend(reversed_frames[:remaining])
        
        return extended[:target_length]
    
    def _build_sequence_tensor(self, frames, optical_flows):
        """
        Build sequence tensor with shape [3, 150, 512, 512]
        
        Channel 0: Grayscale frames
        Channel 1: Optical flow X component
        Channel 2: Optical flow Y component
        
        First frame has zero flow
        """
        # Initialize tensor
        sequence = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        
        # Set grayscale channel (0)
        for i, frame in enumerate(frames):
            sequence[0, i] = torch.from_numpy(frame.astype(np.float32) / 255.0)
        
        # Set optical flow channels (1-2)
        # First frame has zero flow
        for i, flow in enumerate(optical_flows):
            sequence[1, i+1] = torch.from_numpy(flow[:, :, 0])  # X component
            sequence[2, i+1] = torch.from_numpy(flow[:, :, 1])  # Y component
        
        return sequence


import pandas as pd
import os

def create_data_loaders(data_dir, dataset_name, batch_size=2, frame_size=512, 
                       num_frames=150, train_split=0.7, val_split=0.1, num_workers=1,
                       split_file=None, one_hot=False):
    """
    Create train, validation, and test DataLoaders
    
    Args:
        data_dir (str): Root directory containing datasets
        dataset_name (str): Name of dataset
        batch_size (int): Batch size for DataLoader
        frame_size (int): Size to resize frames to
        num_frames (int): Number of frames to extract
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        num_workers (int): Number of workers for DataLoader
        split_file (str): Path to save/load split assignments (default: dataset_name_split.csv)
        one_hot (bool): If True, return one-hot encoded labels instead of binary labels
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Create full dataset
    full_dataset = DenseOpticalFlowDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        frame_size=frame_size,
        num_frames=num_frames,
        one_hot=one_hot
    )
    
    # Set default split file if not provided
    if split_file is None:
        split_file = f"{dataset_name}_split.csv"
    
    dataset_size = len(full_dataset)
    
    # Check if split file exists
    if os.path.exists(split_file):
        print(f"Loading dataset split from {split_file}")
        # Load the split assignments
        split_df = pd.read_csv(split_file)
        
        # Create subset datasets based on saved indices
        train_indices = split_df[split_df['split'] == 'train']['index'].tolist()
        val_indices = split_df[split_df['split'] == 'val']['index'].tolist()
        test_indices = split_df[split_df['split'] == 'test']['index'].tolist()
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)
    else:
        print(f"Creating new dataset split and saving to {split_file}")
        # Calculate split sizes
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Create random indices for the splits
        indices = list(range(dataset_size))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Save the split to CSV
        split_data = []
        for i in range(dataset_size):
            if i in train_indices:
                split = 'train'
            elif i in val_indices:
                split = 'val'
            else:
                split = 'test'
            
            # Save index and corresponding video path for reference
            video_path = full_dataset.video_paths[i]
            split_data.append({
                'index': i, 
                'video_path': video_path,
                'split': split
            })
        
        # Create and save DataFrame
        split_df = pd.DataFrame(split_data)
        split_df.to_csv(split_file, index=False)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset_name = "testdataset"  # Path: data/ensemble with violence and nonviolence subfolders
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=2,
        frame_size=512,
        num_frames=150,
        num_workers=0
    )
    
    # Test the DataLoader
    print("Testing DataLoader...")
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx}:")
        print(f"  Sequences shape: {sequences.shape}")  # Should be (batch_size, 3, 150, 512, 512)
        print(f"  Labels shape: {labels.shape}")        # Should be (batch_size,)
        print(f"  Labels: {labels}")
        
        if batch_idx == 0:  # Only test first batch
            break