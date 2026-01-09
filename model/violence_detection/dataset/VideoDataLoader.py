import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
from PIL import Image
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, data_dir, dataset_name, transform=None, figure_size=244, 
                 seq_length=20, crop_percentage=0.8, crop_dark=None):
        """
        PyTorch Dataset for Violence Detection
        
        Args:
            data_dir (str): Root directory containing datasets
            dataset_name (str): Name of dataset (will be converted to lowercase)
            transform (callable, optional): Optional transform to be applied on frames
            figure_size (int): Size to resize frames to (figure_size x figure_size)
            seq_length (int): Number of frame differences to extract (20)
            crop_percentage (float): Percentage for random cropping (0.8 = 80%)
            crop_dark (tuple): Fixed crop coordinates (x_crop, y_crop) for dark border removal
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.figure_size = figure_size
        self.seq_length = seq_length
        self.crop_percentage = crop_percentage
        self.crop_dark = crop_dark
        
        # Define dataset path
        self.dataset_path = os.path.join(data_dir, self.dataset_name)
        
        # Define corner keys for consistent cropping
        self.corner_keys = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        self.crop_positions = {}  # Store consistent crop positions for each video
        
        self._load_dataset()
        
        # Custom transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((figure_size, figure_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transform
    
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
                    # Pre-determine crop position for consistency
                    self.crop_positions[video_path] = random.choice(self.corner_keys)
        
        # Load violence videos (label = 1)
        violence_path = os.path.join(self.dataset_path, "violence")
        if os.path.exists(violence_path):
            for video_file in os.listdir(violence_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                    video_path = os.path.join(violence_path, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(1)
                    # Pre-determine crop position for consistency
                    self.crop_positions[video_path] = random.choice(self.corner_keys)
        
        print(f"Loaded {len(self.video_paths)} videos from {self.dataset_name}")
        print(f"Nonviolence: {self.labels.count(0)}, Violence: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence with frame differences
        
        Returns:
            sequence (tensor): Shape (seq_length, 3, figure_size, figure_size) - 20 frame differences
            label (tensor): Binary label (0 or 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract 21 consecutive frames and get 20 differences
        frames = self._extract_frames(video_path)
        
        # Convert frames to tensor and apply preprocessing
        sequence = self._process_frames(frames, video_path)
        
        return sequence, torch.tensor(label, dtype=torch.float)
    
    def _extract_frames(self, video_path):
        """Extract 21 random consecutive frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
            # Handle case where video has no frames or invalid frame count
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has {total_frames} frames")
            cap.release()
            # Return dummy black frames
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return [dummy_frame] * 21
        
        if total_frames < 21:
            # If video has less than 21 frames, repeat frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = []
            while len(frames) < 21:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    continue
                frames.append(frame)
            cap.release()
            return frames[:21]
        
        # Randomly select starting frame (ensure we can get 21 consecutive frames)
        start_frame = random.randint(0, total_frames - 21)
        
        # Extract 21 consecutive frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(21):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Ensure we have exactly 21 frames
        while len(frames) < 21:
            frames.extend(frames)  # Repeat last frame if needed
                
        return frames[:21]
    
    def _process_frames(self, frames, video_path):
        """
        Process frames: crop, resize, normalize, and compute differences
        
        Args:
            frames (list): List of 21 frames
            video_path (str): Path to video for consistent cropping
            
        Returns:
            tensor: Shape (seq_length, 3, figure_size, figure_size)
        """
        processed_frames = []
        
        # Get consistent crop position for this video
        crop_corner = self.crop_positions[video_path]
        
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply dark border removal if specified
            if self.crop_dark:
                frame_rgb = self._crop_dark_borders(frame_rgb, self.crop_dark)
            
            # Apply consistent random crop
            frame_rgb = self._crop_img(frame_rgb, crop_corner)
            
            # Resize to target size
            frame_rgb = cv2.resize(frame_rgb, (self.figure_size, self.figure_size))
            
            # Convert to float32 and normalize to [0, 1]
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            frame_rgb = (frame_rgb - np.array(self.mean)) / np.array(self.std)
            
            # Convert to tensor: (H, W, C) -> (C, H, W)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
            processed_frames.append(frame_tensor)
        
        # Compute frame differences (21 frames -> 20 differences)
        frame_diffs = []
        for i in range(len(processed_frames) - 1):
            diff = processed_frames[i] - processed_frames[i + 1]
            frame_diffs.append(diff)
        
        # Stack to create sequence: (seq_length, 3, H, W)
        sequence = torch.stack(frame_diffs).float()
        
        return sequence
    
    def _crop_dark_borders(self, img, crop_coords):
        """Remove dark borders from image"""
        x_crop, y_crop = crop_coords
        h, w = img.shape[:2]
        
        x_start = x_crop
        x_end = w - x_crop
        y_start = y_crop
        y_end = h - y_crop
        
        return img[y_start:y_end, x_start:x_end]
    
    def _crop_img(self, img, corner):
        """Apply consistent random crop based on corner position"""
        h, w = img.shape[:2]
        
        # Calculate crop size
        crop_size = int(min(h, w) * self.crop_percentage)
        
        if corner == "Left_up":
            x_start, y_start = 0, 0
        elif corner == "Right_down":
            x_start = w - crop_size
            y_start = h - crop_size
        elif corner == "Right_up":
            x_start = w - crop_size
            y_start = 0
        elif corner == "Left_down":
            x_start = 0
            y_start = h - crop_size
        else:  # Center
            x_start = (w - crop_size) // 2
            y_start = (h - crop_size) // 2
        
        # Ensure coordinates are within bounds
        x_start = max(0, min(x_start, w - crop_size))
        y_start = max(0, min(y_start, h - crop_size))
        
        x_end = x_start + crop_size
        y_end = y_start + crop_size
        
        return img[y_start:y_end, x_start:x_end]


def create_data_loaders(data_dir, dataset_name, batch_size=2, figure_size=244, 
                       seq_length=20, crop_dark=None, train_split=0.7, 
                       val_split=0.1, num_workers=4):
    """
    Create train, validation, and test DataLoaders
    
    Args:
        data_dir (str): Root directory containing datasets
        dataset_name (str): Name of dataset
        batch_size (int): Batch size for DataLoader
        figure_size (int): Size to resize frames to
        seq_length (int): Number of frame differences
        crop_dark (tuple): Crop coordinates for dark border removal
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        num_workers (int): Number of workers for DataLoader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Create full dataset
    full_dataset = VideoDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        figure_size=figure_size,
        seq_length=seq_length,
        crop_dark=crop_dark
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    # train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = 10
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset_name = "hocky"  # Will be converted to lowercase
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=2,
        figure_size=244,
        seq_length=20,
        crop_dark=(11, 38),  # For hockey dataset
        num_workers=2
    )
    
    # Test the DataLoader
    print("Testing DataLoader...")
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Sequences shape: {sequences.shape}")  # Should be (batch_size, seq_length, 3, H, W)
        print(f"  Labels shape: {labels.shape}")        # Should be (batch_size,)
        print(f"  Labels: {labels}")
        
        if batch_idx == 0:  # Only test first batch
            break
