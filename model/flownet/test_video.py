import os
import cv2
import numpy as np
import torch
import PIL.Image
import math
from tqdm import tqdm
from run import estimate  # Import the estimate function from run.py

def flow_to_image(flow):
    """
    Converts flow into a RGB image.
    Args:
        flow: numpy array of shape (H, W, 2)
    Returns:
        img: numpy array of shape (H, W, 3)
    """
    u = flow[..., 0]
    v = flow[..., 1]
    rad = np.sqrt(u ** 2 + v ** 2)
    rad_max = np.max(rad)
    
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    
    # Convert flow to HSV (Hue, Saturation, Value)
    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    
    # Hue (represents direction)
    img[..., 0] = (np.arctan2(v, u) / (2.0 * np.pi) + 0.5) % 1.0
    
    # Saturation (bright colors)
    img[..., 1] = 1.0
    
    # Value (brightness based on magnitude)
    img[..., 2] = np.clip(rad / rad_max, 0, 1)
    
    # Convert HSV to RGB
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img_rgb

def process_video_liteflownet(input_video_path, output_video_path=None, output_frames_dir=None, 
                             model_type='default', skip_frames=0, show_progress=True):
    """
    Process a video using LiteFlowNet to generate dense optical flow visualization.
    
    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str, optional): Path to save the output flow video. If None, 
                                         generated based on input name
        output_frames_dir (str, optional): Path to save individual flow frames. If None, frames are not saved
        model_type (str): LiteFlowNet model type ('default', 'kitti', or 'sintel')
        skip_frames (int): Number of frames to skip between processing (0 = process every consecutive pair)
        show_progress (bool): Whether to show a progress bar
        
    Returns:
        str: Path to the output flow video
    """
    # Set default output path if not specified
    if output_video_path is None:
        base_name = os.path.splitext(input_video_path)[0]
        output_video_path = f"{base_name}_flow.mp4"
    
    # Create output frames directory if needed
    if output_frames_dir is not None:
        os.makedirs(output_frames_dir, exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame")
    
    # Frame counter
    frame_idx = 0
    processed_count = 0
    
    # Create a progress bar if requested
    if show_progress:
        pbar = tqdm(total=total_frames-1, desc="Processing frames")
    
    # Process video
    while True:
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames if requested
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            if show_progress:
                pbar.update(1)
            continue
            
        # Convert frames to the format expected by LiteFlowNet
        # LiteFlowNet expects RGB float tensors with shape [C, H, W] and values in range [0, 1]
        prev_tensor = torch.FloatTensor(np.ascontiguousarray(
            prev_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        ))
        curr_tensor = torch.FloatTensor(np.ascontiguousarray(
            curr_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        ))
        
        # Compute optical flow using LiteFlowNet
        with torch.no_grad():  # Disable gradient computation for efficiency
            flow_tensor = estimate(prev_tensor, curr_tensor)
        
        # Convert flow to numpy
        flow = flow_tensor.numpy().transpose(1, 2, 0)
        
        # Visualize flow
        flow_img = flow_to_image(flow)
        
        # Write flow image to video
        flow_img_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
        out.write(flow_img_bgr)
        
        # Save flow frame if requested
        if output_frames_dir is not None:
            frame_path = os.path.join(output_frames_dir, f"flow_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, flow_img_bgr)
        
        # Update progress bar
        if show_progress:
            pbar.update(1)
            
        # Update previous frame
        prev_frame = curr_frame.copy()
        processed_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    if show_progress:
        pbar.close()
    
    print(f"Processed {processed_count} frame pairs")
    print(f"Flow video saved to {output_video_path}")
    
    return output_video_path

# Example usage
if __name__ == "__main__":
    # Process a video to generate optical flow visualization
    process_video_liteflownet(
        input_video_path="video/Fight_946.mp4",
        output_video_path="video/Fight_946_flow.mp4",
        output_frames_dir="video/Fight_946_flow_frames",
        model_type="default",
        skip_frames=0  # Process every consecutive pair of frames
    )
    