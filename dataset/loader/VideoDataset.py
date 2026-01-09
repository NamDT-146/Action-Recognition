import os
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from ..transforms.VideoTransforms import VideoTransforms
from transformers import AutoImageProcessor


class VideoDataset(Dataset):
    """
    PyTorch Dataset for Violence Detection using merged VideoTransforms.
    
    Dataset structure:
    - data_dir/dataset_name/
        - nonviolence/
            - video1.mp4, video2.mp4, ...
        - violence/
            - video1.mp4, video2.mp4, ...
    """
    
    def __init__(self, data_dir, dataset_name, figure_size=224, 
                 seq_length=20, crop_percentage=0.8, crop_dark=None, model_name=None):
        """
        Args:
            data_dir (str): Root data directory
            dataset_name (str): Dataset folder name
            figure_size (int): Target frame size
            seq_length (int): Number of frame differences
            crop_percentage (float): Crop percentage for random cropping
            crop_dark (tuple): Coordinates for dark border removal
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.figure_size = figure_size
        self.seq_length = seq_length
        self.crop_percentage = crop_percentage
        self.crop_dark = crop_dark
        
        self.dataset_path = os.path.join(data_dir, self.dataset_name)
        self.corner_keys = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        self.video_paths = []
        self.labels = []
        self.crop_positions = {} 
        
        self._load_dataset()

        # Create transform pipeline
        self.transform = VideoTransforms.get_preprocessing_transform(
            figure_size=self.figure_size,
            seq_length=self.seq_length,
            crop_dark=self.crop_dark,
            crop_percentage=self.crop_percentage,
            crop_corner=None,
            model_name=model_name     
        )
    
    def _load_dataset(self):
        """Load videos from nonviolence and violence folders"""
        for label_name, label_val in [("nonviolence", 0), ("violence", 1)]:
            path = os.path.join(self.dataset_path, label_name)
            print(f"Loading videos from: {path}")
            if os.path.exists(path):
                for video_file in os.listdir(path):
                    if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                        v_path = os.path.join(path, video_file)
                        self.video_paths.append(v_path)
                        self.labels.append(label_val)
                        self.crop_positions[v_path] = random.choice(self.corner_keys)
        
        print(f"Loaded {len(self.video_paths)} videos from {self.dataset_name}")
        if len(self.video_paths) > 0:
            print(f"  Nonviolence: {sum(1 for l in self.labels if l == 0)}")
            print(f"  Violence: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence with frame differences
        
        Returns:
            sequence (tensor): Frame differences (seq_length, 3, figure_size, figure_size)
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
            sequence = torch.zeros(self.seq_length, 3, self.figure_size, self.figure_size)
        
        return sequence, torch.tensor(label, dtype=torch.float)


def create_video_dataset(data_dir="/home/atin-ct3/action_recognition/data",
                         dataset_name="RFW-2000-cleaned",
                         figure_size=224,
                         seq_length=20,
                         crop_percentage=0.8,
                         crop_dark=None,
                         **kwargs):
    """
    Create Video Violence Detection Dataset
    
    Args:
        data_dir (str): Root data directory
        dataset_name (str): Dataset folder name
        figure_size (int): Target frame size
        seq_length (int): Number of frame differences
        crop_percentage (float): Crop percentage for random cropping
        crop_dark (tuple): Coordinates for dark border removal
        **kwargs: Additional arguments to ignore
        
    Returns:
        VideoDataset: Dataset instance
    """
    return VideoDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        figure_size=figure_size,
        seq_length=seq_length,
        crop_percentage=crop_percentage,
        crop_dark=crop_dark
    )


# Example usage
if __name__ == "__main__":
    dataset = VideoDataset(
        data_dir="/home/atin-ct3/action_recognition/data",
        dataset_name="RFW-2000-cleaned",
        figure_size=224,
        seq_length=20
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total videos: {len(dataset)}")
    
    print(f"\nTesting sample access:")
    sequence, label = dataset[0]
    print(f"  Sequence shape: {sequence.shape}")
    print(f"  Label: {label.item()} ({'Violence' if label.item() == 1 else 'Nonviolence'})")

class TimeSformerDataset(Dataset):
    """
    Dataset for TimeSformer fine-tuning.
    
    Dataset structure:
    - data_dir/dataset_name/
        - class1/
            - video1.mp4, video2.mp4, ...
        - class2/
            - video1.mp4, video2.mp4, ...
    """
    
    def __init__(self, data_dir, dataset_name, num_frames=8, 
                 model_name="facebook/timesformer-base-finetuned-k400"):
        """
        Args:
            data_dir: Root data directory
            dataset_name: Dataset folder name
            num_frames: Number of frames to sample
            model_name: HuggingFace model name for processor
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.dataset_path = os.path.join(data_dir, dataset_name)
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Load dataset
        self.video_paths = []
        self.labels = []
        self.class_names = []
        self._load_dataset()
        
        print(f"Loaded {len(self.video_paths)} videos from {self.dataset_name}")
        print(f"Classes: {self.class_names}")
    
    def _load_dataset(self):
        """Load videos from class folders."""
        # Get class folders
        class_folders = sorted([d for d in os.listdir(self.dataset_path) 
                               if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        self.class_names = class_folders
        
        for label_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.dataset_path, class_name)
            
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                    video_path = os.path.join(class_path, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(label_idx)
        
        # Print class distribution
        for idx, class_name in enumerate(self.class_names):
            count = sum(1 for l in self.labels if l == idx)
            print(f"  {class_name}: {count} videos")
    
    def load_video(self, video_path):
        """
        Load and uniformly sample frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames (numpy arrays in RGB)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        
        if not frames:
            # Return dummy frames if loading fails
            print(f"Warning: Could not load {video_path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
        
        # Uniform sampling
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]
        
        return sampled_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get video and label.
        
        Returns:
            pixel_values: Processed frames (num_frames, C, H, W)
            label: Class label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load frames
            frames = self.load_video(video_path)
            
            # Process frames using TimeSformer processor
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return dummy tensor
            pixel_values = torch.zeros(self.num_frames, 3, 224, 224)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)


def create_timesformer_dataset(data_dir, dataset_name, num_frames=8,
                               model_name="facebook/timesformer-base-finetuned-k400", **kwargs):
    """
    Factory function to create TimeSformer dataset.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset folder name
        num_frames: Number of frames to sample
        model_name: HuggingFace model name
        
    Returns:
        TimeSformerDataset instance
    """
    return TimeSformerDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        num_frames=num_frames,
        model_name=model_name
    )