import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from flownet.run import estimate  # Import from liteflownet
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def extract_frames_and_save(video_path, save_dir, frame_size=224, class_id=None, split=None):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if split is not None and class_id is not None:
        save_subdir = os.path.join(save_dir, split, str(class_id))
    elif split is not None:
        save_subdir = os.path.join(save_dir, split)
    elif class_id is not None:
        save_subdir = os.path.join(save_dir, str(class_id))
    else:
        save_subdir = save_dir
        
    os.makedirs(save_subdir, exist_ok=True)
    
    rgb_save_path = os.path.join(save_subdir, f"{video_name}_rgb.npy")
    flow_save_path = os.path.join(save_subdir, f"{video_name}_flow.npy")
    
    if os.path.exists(rgb_save_path) and os.path.exists(flow_save_path):
        print(f"Skipping {video_name} - already processed")
        return rgb_save_path, flow_save_path
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None
    
    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize to frame_size x frame_size
        frame = cv2.resize(frame, (frame_size, frame_size))
        
        # Store in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    
    cap.release()
    
    if len(all_frames) == 0:
        print(f"Error: No frames read from {video_path}")
        return None, None
    
    # Convert to numpy array [T, H, W, C]
    all_frames = np.array(all_frames)
    np.save(rgb_save_path, all_frames)
    optical_flows = []
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in all_frames]
    
    # Compute optical flow between consecutive frames
    for i in tqdm(range(len(gray_frames) - 1), desc=f"Computing flow for {video_name}", leave=False):
        prev_frame = gray_frames[i]
        curr_frame = gray_frames[i + 1]
        
        # Convert to RGB float tensors (LiteFlowNet expects RGB)
        prev_tensor = frame_to_tensor(prev_frame)
        curr_tensor = frame_to_tensor(curr_frame)
        
        # Compute optical flow
        with torch.no_grad():
            flow_tensor = estimate(prev_tensor, curr_tensor)
        
        # Convert to numpy array
        flow = flow_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
        optical_flows.append(flow)
    
    # Add zero flow for first frame to match frame count
    zero_flow = np.zeros((frame_size, frame_size, 2), dtype=np.float32)
    optical_flows = [zero_flow] + optical_flows
    
    # Convert to numpy array [T, H, W, 2]
    optical_flows = np.array(optical_flows)
    
    # Save optical flow (using compressed npz to save space)
    np.save(flow_save_path, optical_flows)  # Instead of savez_compressed    
    
    print(f"Saved: {rgb_save_path} - Shape: {all_frames.shape}")
    print(f"Saved: {flow_save_path} - Shape: {optical_flows.shape}")
    
    return rgb_save_path, flow_save_path

def frame_to_tensor(gray_frame):
    """Convert grayscale frame to RGB tensor for LiteFlowNet"""
    # Repeat grayscale to create RGB
    rgb = np.stack([gray_frame] * 3, axis=-1)
    
    # Convert to float tensor [0,1]
    tensor = torch.FloatTensor(
        np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    ).cuda()
    
    return tensor

def precompute_dataset(data_dir, save_dir, test_size=0.2, frame_size=512, random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all videos
    video_paths = []
    labels = []
    
    nonviolence_path = os.path.join(data_dir, "nonviolence")
    if os.path.exists(nonviolence_path):
        for video_file in os.listdir(nonviolence_path):
            if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                video_paths.append(os.path.join(nonviolence_path, video_file))
                labels.append(0)
    
    violence_path = os.path.join(data_dir, "violence")
    if os.path.exists(violence_path):
        for video_file in os.listdir(violence_path):
            if video_file.endswith(('.avi', '.mp4', '.mpg', '.mov')):
                video_paths.append(os.path.join(violence_path, video_file))
                labels.append(1)
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Total videos: {len(video_paths)}")
    print(f"Train set: {len(train_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    split_info = []
    
    print("Processing training set...")
    for video_path, label in tqdm(zip(train_paths, train_labels), total=len(train_paths)):
        rgb_path, flow_path = extract_frames_and_save(
            video_path, save_dir, frame_size=frame_size, class_id=label, split='train'
        )
        if rgb_path:
            split_info.append({
                'video_path': video_path,
                'rgb_path': rgb_path,
                'flow_path': flow_path,
                'label': label,
                'split': 'train'
            })
    
    print("Processing test set...")
    for video_path, label in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
        rgb_path, flow_path = extract_frames_and_save(
            video_path, save_dir, frame_size=frame_size, class_id=label, split='test'
        )
        if rgb_path:
            split_info.append({
                'video_path': video_path,
                'rgb_path': rgb_path,
                'flow_path': flow_path,
                'label': label,
                'split': 'test'
            })
    
    # Save split info to CSV
    split_df = pd.DataFrame(split_info)
    split_df.to_csv(os.path.join(save_dir, 'split_info.csv'), index=False)
    print(f"Split info saved to {os.path.join(save_dir, 'split_info.csv')}")

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

# Example usage
if __name__ == '__main__':
    # Example usage
    data_dir = "data/ensemble"  # Directory containing violence/nonviolence folders
    save_dir = "data/precomputed_ensemble"  # Directory to save preprocessed data

    # Precompute dataset
    precompute_dataset(
        data_dir=data_dir,
        save_dir=save_dir,
        test_size=0.2,
        frame_size=256
    )
    
    # Example of loading data
    split_info = pd.read_csv(os.path.join(save_dir, 'split_info.csv'))
    
    # Example loading first training video
    train_sample = split_info[split_info['split'] == 'train'].iloc[0]
    
    # Load RGB and flow
    rgb_frames = load_rgb_frames(train_sample['rgb_path'], start=0, num=64)
    flow_frames = load_flow_frames(train_sample['flow_path'], start=0, num=64)
    
    print(f"Loaded RGB frames shape: {rgb_frames.shape}")
    print(f"Loaded flow frames shape: {flow_frames.shape}")