import os
import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ..transforms.VideoTransforms import VideoTransforms


class RFWDataset(Dataset):
    """
    Dataset for RFW-2000-cleaned Violence Detection.
    
    Dataset structure:
    - data_dir/RFW-2000-cleaned/
        - nonviolence/
            - video1.mp4, video2.mp4, ...
        - violence/
            - video1.mp4, video2.mp4, ...
    
    Features:
    - Preserves class ratio in train/test split
    - Creates/loads CSV split file for reproducibility
    - Loads raw video frames without frame differences
    """
    
    def __init__(self, data_dir, split='train', figure_size=224, 
                 num_frames=8, frame_step=1, model_name=None, split_file=None):
        """
        Args:
            data_dir (str): Root data directory
            split (str): 'train', 'val', or 'test'
            figure_size (int): Target frame size
            num_frames (int): Number of frames to extract
            frame_step (int): Step between frames (1=consecutive, 2=skip 1, etc.)
            model_name (str): Model name for specific preprocessing
            split_file (str): Path to CSV split file
        """
        self.data_dir = data_dir
        self.split = split
        self.figure_size = figure_size
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.model_name = model_name
        
        # Set split file path
        if split_file is None:
            self.split_file = os.path.join(data_dir, 'splits.csv')
        else:
            self.split_file = split_file
        
        # Load or create splits
        self.video_paths = []
        self.labels = []
        self.class_names = ['nonviolence', 'violence']
        
        self._load_or_create_splits()
        
        # Create transform pipeline (no frame differences)
        self.transform = VideoTransforms.get_preprocessing_transform(
            figure_size=self.figure_size,
            seq_length=self.num_frames,
            crop_dark=None,
            crop_percentage=0.8,
            crop_corner=None,
            frame_step=self.frame_step,
            model_name=self.model_name
        )
        
        print(f"\n{split.upper()} set loaded:")
        print(f"  Total videos: {len(self.video_paths)}")
        print(f"  Nonviolence: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Violence: {sum(1 for l in self.labels if l == 1)}")
    
    def _scan_dataset(self):
        """
        Scan dataset directory and collect all video paths and labels.
        
        Returns:
            tuple: (video_paths, labels)
        """
        video_paths = []
        labels = []
        
        for label_name, label_val in [("nonviolence", 0), ("violence", 1)]:
            class_path = os.path.join(self.data_dir, label_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
            
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov', '.MP4', '.AVI')):
                    video_path = os.path.join(class_path, video_file)
                    video_paths.append(video_path)
                    labels.append(label_val)
        
        return video_paths, labels
    
    def _create_splits(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        Create train/val/test splits with stratification.
        
        Args:
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with columns ['video_path', 'label', 'split']
        """
        print(f"\nCreating new splits for {self.data_dir}...")
        
        # Scan dataset
        all_videos, all_labels = self._scan_dataset()
        
        if len(all_videos) == 0:
            raise ValueError(f"No videos found in {self.data_dir}")
        
        print(f"Found {len(all_videos)} videos:")
        print(f"  Nonviolence: {sum(1 for l in all_labels if l == 0)}")
        print(f"  Violence: {sum(1 for l in all_labels if l == 1)}")
        
        # First split: train + val vs test
        test_ratio = 1.0 - train_ratio - val_ratio
        train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
            all_videos, all_labels,
            test_size=test_ratio,
            stratify=all_labels,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            train_val_videos, train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=random_state
        )
        
        # Create DataFrame
        data = []
        for video, label in zip(train_videos, train_labels):
            data.append({'video_path': video, 'label': label, 'split': 'train'})
        for video, label in zip(val_videos, val_labels):
            data.append({'video_path': video, 'label': label, 'split': 'val'})
        for video, label in zip(test_videos, test_labels):
            data.append({'video_path': video, 'label': label, 'split': 'test'})
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(self.split_file, index=False)
        print(f"\nSplits saved to {self.split_file}")
        print(f"  Train: {len(train_videos)}")
        print(f"  Val: {len(val_videos)}")
        print(f"  Test: {len(test_videos)}")
        
        return df
    
    def _load_or_create_splits(self):
        """Load existing splits or create new ones."""
        if os.path.exists(self.split_file):
            print(f"\nLoading existing splits from {self.split_file}")
            df = pd.read_csv(self.split_file)
        else:
            df = self._create_splits()
        
        # Filter by split
        split_df = df[df['split'] == self.split]
        
        self.video_paths = split_df['video_path'].tolist()
        self.labels = split_df['label'].tolist()
        
        # Verify files exist
        valid_indices = []
        for idx, video_path in enumerate(self.video_paths):
            if os.path.exists(video_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: {video_path} does not exist")
        
        self.video_paths = [self.video_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence.
        
        Returns:
            sequence (tensor): Raw frames (num_frames, 3, figure_size, figure_size)
            label (tensor): Binary label (0 or 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract and process frames
            sequence = self.transform(video_path)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy sequence on error
            sequence = torch.zeros(self.num_frames, 3, self.figure_size, self.figure_size)
        
        return sequence, torch.tensor(label, dtype=torch.long)


def get_rfw_dataset(data_dir="data/RFW-2000-cleaned", 
                    split='train',
                    figure_size=224,
                    num_frames=8,
                    frame_step=1,
                    model_name=None,
                    **kwargs):
    """
    Factory function to create RFW dataset.
    
    Args:
        frame_step (int): Step between frames
        
    Returns:
        RFWDataset: Dataset instance
    """
    return RFWDataset(
        data_dir=data_dir,
        split=split,
        figure_size=figure_size,
        num_frames=num_frames,
        frame_step=frame_step,
        model_name=model_name
    )


def create_rfw_dataloaders(data_dir="data/RFW-2000-cleaned",
                           batch_size=4,
                           figure_size=224,
                           num_frames=8,
                           frame_step=1,
                           model_name=None,
                           num_workers=0):
    """
    Create train, val, and test dataloaders for RFW dataset.
    
    Args:
        frame_step (int): Step between frames
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = get_rfw_dataset(
        data_dir=data_dir,
        split='train',
        figure_size=figure_size,
        num_frames=num_frames,
        frame_step=frame_step,
        model_name=model_name
    )
    
    val_dataset = get_rfw_dataset(
        data_dir=data_dir,
        split='val',
        figure_size=figure_size,
        num_frames=num_frames,
        frame_step=frame_step,
        model_name=model_name
    )
    
    test_dataset = get_rfw_dataset(
        data_dir=data_dir,
        split='test',
        figure_size=figure_size,
        num_frames=num_frames,
        frame_step=frame_step,
        model_name=model_name
    )
    
    # Create dataloaders
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
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Test dataset creation
    print("=" * 60)
    print("Testing RFW Dataset")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = create_rfw_dataloaders(
        data_dir="data/RFW-2000-cleaned",
        batch_size=2,
        figure_size=224,
        num_frames=16,
        num_workers=0
    )
    
    print("\n" + "=" * 60)
    print("Testing batch loading")
    print("=" * 60)
    
    # Test one batch from each loader
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        batch_videos, batch_labels = next(iter(loader))
        print(f"\n{name} batch:")
        print(f"  Videos shape: {batch_videos.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Labels: {batch_labels.tolist()}")