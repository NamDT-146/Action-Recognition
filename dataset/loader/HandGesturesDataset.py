import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from ..transforms.VideoTransforms import VideoTransforms


class HandGesturesDataset(Dataset):
    """
    PyTorch Dataset for Hand Gestures Recognition.
    
    Dataset structure:
    - data_dir/hand-gestures/
        - hand_gestures.csv (with columns: set_id, one, four, small, fist, me)
        - files/
            - 0/one.mp4, 0/four.mp4, 0/small.mp4, 0/fist.mp4, 0/me.mp4
            - 1/one.mp4, 1/four.mp4, 1/small.mp4, 1/fist.mp4, 1/me.mp4
            - ... and so on
    """
    
    # Gesture classes
    GESTURE_CLASSES = {
        'one': 0,
        'four': 1,
        'small': 2,
        'fist': 3,
        'me': 4
    }
    
    CLASS_NAMES = {v: k for k, v in GESTURE_CLASSES.items()}
    
    def __init__(self, data_dir, dataset_name='hand-gestures', figure_size=224, 
                 seq_length=20, crop_percentage=0.8, crop_dark=None, 
                 gesture_filter=None, set_filter=None):
        """
        Args:
            data_dir (str): Root data directory
            dataset_name (str): Dataset folder name (default: 'hand-gestures')
            figure_size (int): Target frame size
            seq_length (int): Number of frame differences
            crop_percentage (float): Crop percentage for random cropping
            crop_dark (tuple): Coordinates for dark border removal
            gesture_filter (list): Filter specific gestures (e.g., ['one', 'four'])
                                   If None, all gestures are included
            set_filter (list): Filter specific set IDs (e.g., [0, 1, 2])
                              If None, all sets are included
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.figure_size = figure_size
        self.seq_length = seq_length
        self.crop_percentage = crop_percentage
        self.crop_dark = crop_dark
        self.gesture_filter = gesture_filter if gesture_filter is not None else list(self.GESTURE_CLASSES.keys())
        self.set_filter = set_filter
        
        self.dataset_path = os.path.join(data_dir, dataset_name)
        self.csv_path = os.path.join(self.dataset_path, 'hand_gestures.csv')
        
        self.corner_keys = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        # Data structures
        self.video_paths = []
        self.labels = []
        self.set_ids = []
        self.crop_positions = {}
        
        # Load dataset from CSV
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        print(f"Loading Hand Gestures Dataset from {self.dataset_path}")
        print(f"Found {len(df)} sets in CSV")
        
        # Iterate through CSV rows
        for idx, row in df.iterrows():
            set_id = row['set_id']
            
            # Apply set filter if specified
            if self.set_filter is not None and set_id not in self.set_filter:
                continue
            
            # Iterate through each gesture column
            for gesture_name in self.GESTURE_CLASSES.keys():
                # Skip if gesture is not in filter
                if gesture_name not in self.gesture_filter:
                    continue
                
                # Get relative path from CSV
                relative_path = row[gesture_name]
                
                # Construct full path
                video_path = os.path.join(self.dataset_path, relative_path)
                
                # Check if file exists
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    continue
                
                # Add to dataset
                self.video_paths.append(video_path)
                self.labels.append(self.GESTURE_CLASSES[gesture_name])
                self.set_ids.append(set_id)
                self.crop_positions[video_path] = random.choice(self.corner_keys)
        
        print(f"Successfully loaded {len(self.video_paths)} videos")
        print(f"Gesture distribution:")
        for gesture_name, gesture_id in self.GESTURE_CLASSES.items():
            count = sum(1 for l in self.labels if l == gesture_id)
            if count > 0:
                print(f"  {gesture_name}: {count} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence with frame differences
        
        Returns:
            sequence (tensor): Frame differences (seq_length, 3, figure_size, figure_size)
            label (tensor): Gesture class ID (0-4)
            metadata (dict): Additional metadata (set_id, gesture_name, video_path)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        set_id = self.set_ids[idx]
        crop_corner = self.crop_positions[video_path]
        
        # Get gesture name
        gesture_name = self.CLASS_NAMES[label]
        
        # Create transform pipeline
        transform = VideoTransforms.get_preprocessing_transform(
            figure_size=self.figure_size,
            seq_length=self.seq_length,
            crop_dark=self.crop_dark,
            crop_percentage=self.crop_percentage,
            crop_corner=crop_corner
        )
        
        try:
            # Extract and process frames
            sequence = transform(video_path)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy sequence on error
            sequence = torch.zeros(self.seq_length, 3, self.figure_size, self.figure_size)
        
        # Metadata
        metadata = {
            'set_id': set_id,
            'gesture_name': gesture_name,
            'video_path': video_path
        }
        
        return sequence, torch.tensor(label, dtype=torch.long), metadata
    
    @staticmethod
    def get_gesture_id(gesture_name):
        """Get gesture ID from gesture name"""
        if gesture_name.lower() in HandGesturesDataset.GESTURE_CLASSES:
            return HandGesturesDataset.GESTURE_CLASSES[gesture_name.lower()]
        raise ValueError(f"Unknown gesture: {gesture_name}")
    
    @staticmethod
    def get_gesture_name(gesture_id):
        """Get gesture name from gesture ID"""
        if gesture_id in HandGesturesDataset.CLASS_NAMES:
            return HandGesturesDataset.CLASS_NAMES[gesture_id]
        raise ValueError(f"Unknown gesture ID: {gesture_id}")
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_videos': len(self),
            'total_sets': len(set(self.set_ids)),
            'gestures': {},
            'sets_per_gesture': {}
        }
        
        for gesture_name, gesture_id in self.GESTURE_CLASSES.items():
            count = sum(1 for l in self.labels if l == gesture_id)
            set_count = len(set(sid for i, sid in enumerate(self.set_ids) if self.labels[i] == gesture_id))
            stats['gestures'][gesture_name] = count
            stats['sets_per_gesture'][gesture_name] = set_count
        
        return stats


def create_hand_gestures_dataset(data_dir="/home/atin-ct3/action_recognition/data",
                                 dataset_name="hand-gestures",
                                 figure_size=224,
                                 seq_length=20,
                                 crop_percentage=0.8,
                                 crop_dark=None,
                                 gesture_filter=None,
                                 set_filter=None,
                                 **kwargs):
    """
    Create Hand Gestures Dataset
    
    Args:
        data_dir (str): Root data directory
        dataset_name (str): Dataset folder name
        figure_size (int): Target frame size
        seq_length (int): Number of frame differences
        crop_percentage (float): Crop percentage for random cropping
        crop_dark (tuple): Coordinates for dark border removal
        gesture_filter (list): Filter specific gestures
        set_filter (list): Filter specific set IDs
        **kwargs: Additional arguments to ignore
        
    Returns:
        HandGesturesDataset: Dataset instance
    """
    return HandGesturesDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        figure_size=figure_size,
        seq_length=seq_length,
        crop_percentage=crop_percentage,
        crop_dark=crop_dark,
        gesture_filter=gesture_filter,
        set_filter=set_filter
    )


# Example usage
if __name__ == "__main__":
    dataset = HandGesturesDataset(
        data_dir="/home/atin-ct3/action_recognition/data",
        dataset_name="hand-gestures",
        figure_size=224,
        seq_length=20
    )
    
    print(f"\nDataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nTesting sample access:")
    sequence, label, metadata = dataset[0]
    print(f"  Sequence shape: {sequence.shape}")
    print(f"  Label: {label} ({dataset.CLASS_NAMES[label.item()]})")
    print(f"  Metadata: {metadata}")