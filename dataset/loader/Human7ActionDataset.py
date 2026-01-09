import os
import torch
import random
from torch.utils.data import Dataset
from ..transforms.VideoTransforms import VideoTransforms


class Human7ActionDataset(Dataset):
    """
    PyTorch Dataset for Human 7 Action Recognition.
    
    Dataset structure:
    - data_dir/human7action/
        - train/
            - Fall Down/
                - video1.avi, video2.avi, ...
            - Lying Down/
                - video1.avi, video2.avi, ...
            - Sit down/
                - video1.avi, video2.avi, ...
            - Sitting/
                - video1.avi, video2.avi, ...
            - Stand up/
                - video1.avi, video2.avi, ...
            - Standing/
                - video1.avi, video2.avi, ...
            - Walking/
                - video1.avi, video2.avi, ...
        - test/
            - (same structure as train)
    """
    
    # Action classes
    ACTION_CLASSES = {
        'Fall Down': 0,
        'Lying Down': 1,
        'Sit down': 2,
        'Sitting': 3,
        'Stand up': 4,
        'Standing': 5,
        'Walking': 6
    }
    
    CLASS_NAMES = {v: k for k, v in ACTION_CLASSES.items()}
    
    def __init__(self, data_dir, dataset_name='human7action', split='train', 
                 figure_size=224, seq_length=20, crop_percentage=0.8, crop_dark=None,
                 action_filter=None, model_name=None):
        """
        Args:
            data_dir (str): Root data directory
            dataset_name (str): Dataset folder name (default: 'human7action')
            split (str): 'train' or 'test'
            figure_size (int): Target frame size
            seq_length (int): Number of frame differences
            crop_percentage (float): Crop percentage for random cropping
            crop_dark (tuple): Coordinates for dark border removal
            action_filter (list): Filter specific actions (e.g., ['Walking', 'Standing'])
                                 If None, all actions are included
        """
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.figure_size = figure_size
        self.seq_length = seq_length
        self.crop_percentage = crop_percentage
        self.crop_dark = crop_dark
        self.action_filter = action_filter if action_filter is not None else list(self.ACTION_CLASSES.keys())
        
        self.dataset_path = os.path.join(data_dir, dataset_name, split)
        self.corner_keys = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        # Data structures
        self.video_paths = []
        self.labels = []
        self.actions = []
        self.crop_positions = {}
        
        # Load dataset
        self._load_dataset()

        self.transform = VideoTransforms.get_preprocessing_transform(
            figure_size=self.figure_size,
            seq_length=self.seq_length,
            crop_dark=self.crop_dark,
            crop_percentage=self.crop_percentage,
            crop_corner=self.corner_keys,
            model_name=model_name     
        )
    
    def _load_dataset(self):
        """Load dataset from directory structure"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        print(f"Loading Human7Action Dataset from {self.dataset_path}")
        
        # Iterate through action classes
        for action_name, action_id in self.ACTION_CLASSES.items():
            # Skip if action is not in filter
            if action_name not in self.action_filter:
                continue
            
            action_path = os.path.join(self.dataset_path, action_name)
            
            # Check if action folder exists
            if not os.path.exists(action_path):
                print(f"Warning: Action folder not found: {action_path}")
                continue
            
            # Load all videos in this action folder
            video_count = 0
            for video_file in os.listdir(action_path):
                if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                    video_path = os.path.join(action_path, video_file)
                    
                    # Check if file exists and is readable
                    if not os.path.exists(video_path):
                        print(f"Warning: Video file not found: {video_path}")
                        continue
                    
                    # Add to dataset
                    self.video_paths.append(video_path)
                    self.labels.append(action_id)
                    self.actions.append(action_name)
                    self.crop_positions[video_path] = random.choice(self.corner_keys)
                    video_count += 1
            
            print(f"  {action_name}: {video_count} videos")
        
        print(f"Successfully loaded {len(self.video_paths)} videos for {self.split} split")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a video sequence with frame differences
        
        Returns:
            sequence (tensor): Frame differences (seq_length, 3, figure_size, figure_size)
            label (tensor): Action class ID (0-6)
            metadata (dict): Additional metadata (action_name, video_path, split)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        action_name = self.actions[idx]
        
        try:
            # Extract and process frames
            sequence = self.transform(video_path)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy sequence on error
            sequence = torch.zeros(self.seq_length, 3, self.figure_size, self.figure_size)
        
        # Metadata
        metadata = {
            'action_name': action_name,
            'video_path': video_path,
            'split': self.split
        }
        
        # print("Sequence shape:", sequence.shape)

        return sequence, torch.tensor(label, dtype=torch.long), metadata
    
    @staticmethod
    def get_action_id(action_name):
        """Get action ID from action name"""
        if action_name in Human7ActionDataset.ACTION_CLASSES:
            return Human7ActionDataset.ACTION_CLASSES[action_name]
        raise ValueError(f"Unknown action: {action_name}")
    
    @staticmethod
    def get_action_name(action_id):
        """Get action name from action ID"""
        if action_id in Human7ActionDataset.CLASS_NAMES:
            return Human7ActionDataset.CLASS_NAMES[action_id]
        raise ValueError(f"Unknown action ID: {action_id}")
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_videos': len(self),
            'split': self.split,
            'actions': {}
        }
        
        for action_name, action_id in self.ACTION_CLASSES.items():
            count = sum(1 for l in self.labels if l == action_id)
            stats['actions'][action_name] = count
        
        return stats


def create_human7action_dataset(data_dir="/home/atin-ct3/action_recognition/data",
                                dataset_name="human7action",
                                split='train',
                                figure_size=224,
                                seq_length=20,
                                crop_percentage=0.8,
                                crop_dark=None,
                                action_filter=None,
                                model_name=None,
                                **kwargs):
    """
    Create Human7Action Dataset
    
    Args:
        data_dir (str): Root data directory
        dataset_name (str): Dataset folder name
        split (str): 'train' or 'test'
        figure_size (int): Target frame size
        seq_length (int): Number of frame differences
        crop_percentage (float): Crop percentage for random cropping
        crop_dark (tuple): Coordinates for dark border removal
        action_filter (list): Filter specific actions
        **kwargs: Additional arguments to ignore
        
    Returns:
        Human7ActionDataset: Dataset instance
    """
    return Human7ActionDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
        figure_size=figure_size,
        seq_length=seq_length,
        crop_percentage=crop_percentage,
        crop_dark=crop_dark,
        action_filter=action_filter,
        model_name=model_name
    )


# Example usage
if __name__ == "__main__":
    # Test train split
    print("=" * 80)
    print("TRAIN SPLIT")
    print("=" * 80)
    train_dataset = Human7ActionDataset(
        data_dir="/home/atin-ct3/action_recognition/data",
        dataset_name="human7action",
        split='train',
        figure_size=224,
        seq_length=20
    )
    
    print(f"\nTrain Dataset Statistics:")
    stats = train_dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if len(train_dataset) > 0:
        print(f"\nTesting sample access:")
        sequence, label, metadata = train_dataset[0]
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Label: {label} ({train_dataset.CLASS_NAMES[label.item()]})")
        print(f"  Metadata: {metadata}")
    
    # Test test split
    print("\n" + "=" * 80)
    print("TEST SPLIT")
    print("=" * 80)
    test_dataset = Human7ActionDataset(
        data_dir="/home/atin-ct3/action_recognition/data",
        dataset_name="human7action",
        split='test',
        figure_size=224,
        seq_length=20
    )
    
    print(f"\nTest Dataset Statistics:")
    stats = test_dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if len(test_dataset) > 0:
        print(f"\nTesting sample access:")
        sequence, label, metadata = test_dataset[0]
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Label: {label} ({test_dataset.CLASS_NAMES[label.item()]})")
        print(f"  Metadata: {metadata}")