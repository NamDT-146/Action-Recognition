import torch
from torch.utils.data import DataLoader, random_split
from .loader.VideoDataset import VideoDataset, create_video_dataset, create_timesformer_dataset
from .loader.HandGesturesDataset import HandGesturesDataset, create_hand_gestures_dataset
from .loader.Human7ActionDataset import create_human7action_dataset
from .loader.RWFDataset import get_rfw_dataset, create_rfw_dataloaders

# Dataset registry
DATASET_REGISTRY = {
    'RFW-2000-cleaned': create_video_dataset,
    'rfw': get_rfw_dataset,  # New RFW dataset with splits
    'hockey': create_video_dataset,
    'movies': create_video_dataset,
    'hand-gestures': create_hand_gestures_dataset,
    'human7action': create_human7action_dataset
}

def get_data_loader(data_dir, dataset_name, batch_size=4, figure_size=224,
                    seq_length=20, crop_dark=None, num_workers=4,
                    train_split=0.7, val_split=0.15, gesture_filter=None,
                    set_filter=None, model_name='LSTM_CNN', num_frames=8, 
                    frame_step=1, **kwargs):
    """
    Universal DataLoader creator for all datasets.
    
    Args:
        frame_step (int): Step between frames (1=consecutive, 2=skip 1, etc.)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Special handling for RFW dataset with CSV splits
    if dataset_name == 'rfw':
        return create_rfw_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            figure_size=figure_size,
            num_frames=num_frames,
            frame_step=frame_step,
            model_name=model_name,
            num_workers=num_workers
        )
    
    if model_name == 'TimeSformer':
        # Use TimeSformer dataset
        full_dataset = create_timesformer_dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            num_frames=num_frames
        )
    # Check if dataset is registered
    elif dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    # Get dataset creator function
    create_dataset_fn = DATASET_REGISTRY[dataset_name]
    
    # Prepare dataset-specific kwargs
    dataset_kwargs = {
        'data_dir': data_dir,
        'dataset_name': dataset_name,
        'figure_size': figure_size,
        'seq_length': seq_length,
        'crop_dark': crop_dark,
        'model_name': model_name
    }
    
    # Add optional parameters for hand gestures dataset
    if dataset_name == 'hand-gestures':
        dataset_kwargs['gesture_filter'] = gesture_filter
        dataset_kwargs['set_filter'] = set_filter
    
    # Merge additional kwargs
    dataset_kwargs.update(kwargs)
    
    # Create dataset
    dataset = create_dataset_fn(**dataset_kwargs)
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No videos found in {data_dir}/{dataset_name}")

    # Calculate split sizes
    test_split = 1.0 - train_split - val_split
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {dataset_size}")
    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    return train_loader, val_loader, test_loader


def get_available_datasets():
    """Get list of available datasets"""
    return list(DATASET_REGISTRY.keys())


def register_dataset(dataset_name, create_fn):
    """Register a new dataset creator function"""
    DATASET_REGISTRY[dataset_name] = create_fn