import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
from flownet.run import estimate  # Import from liteflownet

class VideoDataset(Dataset):
    def __init__(self, data_dir, dataset_name, transform=None, figure_size=244, 
                 seq_length=32, crop_percentage=0.8, crop_dark=None):
        """
        PyTorch Dataset for Violence Detection using Optical Flow
        
        Args:
            data_dir (str): Root directory containing datasets
            dataset_name (str): Name of dataset (will be converted to lowercase)
            transform (callable, optional): Optional transform to be applied on frames
            figure_size (int): Size to resize frames to (figure_size x figure_size)
            seq_length (int): Number of optical flows to extract (20)
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
        Get a video sequence with optical flow
        
        Returns:
            sequence (tensor): Shape (3, seq_length, figure_size, figure_size)
                Channel 0: X component of optical flow
                Channel 1: Y component of optical flow
                Channel 2: Magnitude of optical flow
            label (tensor): Binary label (0 or 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        # Compute optical flow and build sequence tensor with magnitude
        with torch.no_grad():
            sequence = self._compute_optical_flow(frames, video_path)
        
        # Return label as tensor
        return sequence, torch.tensor(label, dtype=torch.float)
    
    def _extract_frames(self, video_path):
        """Extract consecutive frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle case where video has no frames or invalid frame count
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has {total_frames} frames")
            cap.release()
            # Return dummy black frames
            dummy_frame = np.zeros((self.figure_size, self.figure_size, 3), dtype=np.uint8)
            return [dummy_frame] * (self.seq_length + 1)
        
        if total_frames < (self.seq_length + 1):
            # If video has less than needed frames, repeat frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = []
            while len(frames) < (self.seq_length + 1):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    continue
                frames.append(frame)
            cap.release()
            return frames[:(self.seq_length + 1)]
        
        # Randomly select starting frame
        start_frame = random.randint(0, total_frames - (self.seq_length + 1))
        
        # Extract consecutive frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(self.seq_length + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Ensure we have enough frames
        while len(frames) < (self.seq_length + 1):
            frames.extend(frames)
                
        return frames[:(self.seq_length + 1)]
    
    def _frame_to_tensor(self, frame, crop_corner):
        """Convert frame to RGB tensor for LiteFlowNet"""
        # Process frame
        if self.crop_dark:
            frame = self._crop_dark_borders(frame, self.crop_dark)
        
        # Apply consistent random crop
        frame = self._crop_img(frame, crop_corner)
        
        # Resize to target size
        frame = cv2.resize(frame, (self.figure_size, self.figure_size))
        
        # Convert to float tensor [0,1]
        tensor = torch.FloatTensor(
            np.ascontiguousarray(frame.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        )
        
        return tensor
    
    def _compute_optical_flow(self, frames, video_path):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames (list): List of frames
            video_path (str): Path to video for consistent cropping
            
        Returns:
            tensor: Shape (3, seq_length, figure_size, figure_size)
                   Channel 0: X component of optical flow
                   Channel 1: Y component of optical flow
                   Channel 2: Magnitude of optical flow
        """
        # Get consistent crop position for this video
        crop_corner = self.crop_positions.get(video_path, random.choice(self.corner_keys))
        
        # Initialize output tensor
        flow_sequence = torch.zeros(self.seq_length, 3, self.figure_size, self.figure_size)
        
        # Process each pair of consecutive frames
        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            curr_frame = frames[i + 1]
            
            # Convert to RGB (LiteFlowNet expects RGB)
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensors for LiteFlowNet using fixed method
            prev_tensor = self._frame_to_tensor(prev_frame, crop_corner)
            curr_tensor = self._frame_to_tensor(curr_frame, crop_corner)
            
            # Compute optical flow
            with torch.no_grad():
                flow_tensor = estimate(prev_tensor, curr_tensor)
            
            # Convert to numpy array
            flow = flow_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # Normalize flow values to [0, 1] range
            flow_max = 20.0  # Define maximum expected flow value
            flow = np.clip(flow, -flow_max, flow_max)
            flow = (flow + flow_max) / (2 * flow_max)  # Map from [-max_flow, max_flow] to [0, 1]
            
            # Compute magnitude (already normalized between 0-1)
            magnitude = np.sqrt(
                ((flow[:,:,0] - 0.5) * 2)**2 + 
                ((flow[:,:,1] - 0.5) * 2)**2
            ) / np.sqrt(2)  # Normalize by √2 for max possible magnitude
            
            # Assign to output tensor
            flow_sequence[i, 0] = torch.from_numpy(flow[:, :, 0])  # X component
            flow_sequence[i, 1] = torch.from_numpy(flow[:, :, 1])  # Y component
            flow_sequence[i, 2] = torch.from_numpy(magnitude)      # Magnitude

        # print(flow_sequence.shape)
        return flow_sequence
    
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
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
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