"""
MIL-based Violence Detection Inference.

This implements the two-stage inference pipeline:
1. X3D extracts 432-dim features from video segments
2. MIL Adapter (LSTM/MLP/Conv1D) predicts from sliding window of 5 features

The logic follows the note.txt architecture:
- X3D: Pre-trained feature extractor (frozen)
- LSTM Adapter: Trained on OCC setting for temporal aggregation
"""

import os
import yaml
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import get_model
from model.Adapter.adapter import get_adapter


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_x3d_model(config_path, checkpoint_path, device):
    """
    Load X3D model for feature extraction.
    
    Args:
        config_path: Path to X3D config YAML
        checkpoint_path: Path to X3D checkpoint
        device: torch.device
        
    Returns:
        X3D model in eval mode
    """
    print(f"Loading X3D model...")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Create model
    model = get_model(**config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ X3D model loaded successfully!")
    
    return model


def load_adapter(adapter_path, input_dim=432, device='cpu'):
    """
    Load MIL adapter model.
    
    Args:
        adapter_path: Path to adapter checkpoint
        input_dim: Input feature dimension
        device: torch.device
        
    Returns:
        Adapter model in eval mode
    """
    print(f"\nLoading MIL Adapter...")
    print(f"  Checkpoint: {adapter_path}")
    
    # Infer adapter type from path
    adapter_path = Path(adapter_path)
    parent_name = adapter_path.parent.name.lower()
    
    if 'mlp' in parent_name:
        adapter_type = 'mlp'
    elif 'lstm' in parent_name:
        adapter_type = 'lstm'
    elif 'conv1d' in parent_name:
        adapter_type = 'conv1d'
    else:
        raise ValueError(f"Cannot infer adapter type from path: {adapter_path}")
    
    print(f"  Detected adapter type: {adapter_type.upper()}")
    
    # Load checkpoint to get config
    checkpoint = torch.load(adapter_path, map_location=device)
    
    # Create adapter (use default hyperparams from training)
    adapter = get_adapter(
        adapter_type=adapter_type,
        input_dim=input_dim,
        hidden_dim=32,  # Default from training
        dropout=0.6
    )
    
    # Load weights
    adapter.load_state_dict(checkpoint['model_state_dict'])
    adapter = adapter.to(device)
    adapter.eval()
    
    print(f"  ✓ Adapter loaded successfully!")
    
    return adapter, adapter_type


def extract_segment_frames(video_path, start_frame, num_frames=16, frame_step=2, img_size=224):
    """
    Extract frames for a single X3D segment.
    
    Args:
        video_path: Path to video
        start_frame: Starting frame index
        num_frames: Number of frames to extract
        frame_step: Step between frames
        img_size: Target image size
        
    Returns:
        torch.Tensor: Preprocessed segment of shape (1, 3, T, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        for i in range(num_frames):
            frame_idx = start_frame + i * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Duplicate last frame if we run out
                if frames:
                    frames.append(frames[-1])
                else:
                    raise IOError(f"Failed to read frame {frame_idx}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame_resized = cv2.resize(
                frame_rgb,
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            
            frames.append(frame_resized)
    finally:
        cap.release()
    
    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    frames_array = np.array(frames, dtype=np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)  # Correct shape (C, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames_tensor = (frames_tensor - mean) / std
    
    # Add batch dimension
    return frames_tensor.unsqueeze(0)  # (1, 3, T, H, W)


def extract_x3d_features_from_video(video_path, x3d_model, device, 
                                    num_frames=16, frame_step=2, 
                                    inference_step=15, img_size=224):
    """
    Extract 432-dim features from entire video using X3D.
    
    Args:
        video_path: Path to video
        x3d_model: X3D model
        device: torch.device
        num_frames: Frames per segment
        frame_step: Step between frames within segment
        inference_step: Step between consecutive segments
        img_size: Image size
        
    Returns:
        features: List of feature tensors (each 432-dim)
        segment_start_frames: List of starting frame indices
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Calculate segment span
    segment_span = (num_frames - 1) * frame_step + 1
    
    print(f"\nExtracting X3D features:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"  Segment config: {num_frames} frames × step {frame_step} = {segment_span} frames span")
    print(f"  Inference step: {inference_step}")
    
    # Calculate segment positions
    segment_starts = list(range(0, total_frames - segment_span + 1, inference_step))
    
    print(f"  Number of segments: {len(segment_starts)}")
    
    # Extract features
    features = []
    segment_start_frames = []
    
    x3d_model.eval()
    
    for start_frame in tqdm(segment_starts, desc="  Extracting features"):
        try:
            # Extract segment
            segment_tensor = extract_segment_frames(
                video_path, start_frame, num_frames, frame_step, img_size
            )
            segment_tensor = segment_tensor.to(device)
            
            # Extract features using X3D
            with torch.no_grad():
                feat = x3d_model.extract_features(segment_tensor)  # (1, 432)
            
            features.append(feat.cpu().squeeze(0))  # (432,)
            segment_start_frames.append(start_frame)
            
        except Exception as e:
            print(f"  Warning: Failed to extract segment at frame {start_frame}: {e}")
            continue
    
    print(f"  ✓ Extracted {len(features)} feature vectors (432-dim each)")
    
    return features, segment_start_frames


def predict_with_sliding_window(features, adapter, device, window_size=5):
    """
    Predict using MIL adapter with sliding window.
    
    Args:
        features: List of feature tensors (each 432-dim)
        adapter: MIL adapter model
        device: torch.device
        window_size: Size of sliding window
        
    Returns:
        predictions: List of (is_violence, confidence) tuples
        prediction_indices: List of feature indices where prediction was made
    """
    print(f"\nRunning MIL adapter predictions:")
    print(f"  Window size: {window_size}")
    print(f"  Total features: {len(features)}")
    
    if len(features) < window_size:
        print(f"  Warning: Not enough features ({len(features)} < {window_size})")
        print(f"           Padding with zeros...")
        
        # Pad with zeros if not enough features
        padded_features = features + [torch.zeros_like(features[0])] * (window_size - len(features))
        features = padded_features
    
    predictions = []
    prediction_indices = []
    
    adapter.eval()
    
    # Sliding window
    for i in range(len(features) - window_size + 1):
        # Get window
        window = features[i:i+window_size]  # List of 5 × (432,)
        
        # Stack to (1, 5, 432)
        window_tensor = torch.stack(window).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            scores = adapter(window_tensor)  # (1, 5)
            
            # Take max score across segments
            max_score, _ = scores.max(dim=1)
            max_score = max_score.item()
        
        # Threshold at 0.5
        is_violence = max_score > 0.5
        confidence = max_score
        
        predictions.append((is_violence, confidence))
        prediction_indices.append(i + window_size - 1)  # Prediction made at last frame of window
    
    print(f"  ✓ Made {len(predictions)} predictions")
    
    return predictions, prediction_indices


def draw_status_on_frame(frame, status, confidence, frame_idx, fps, 
                         segment_start=None, segment_end=None):
    """
    Draw prediction status on frame.
    
    Args:
        frame: Frame in BGR format
        status: Status string ("Initializing", "VIOLENCE", "NON-VIOLENCE")
        confidence: Prediction confidence (or None during initialization)
        frame_idx: Current frame index
        fps: Video FPS
        segment_start: Start frame of segment (if applicable)
        segment_end: End frame of segment (if applicable)
        
    Returns:
        Annotated frame
    """
    frame = frame.copy()
    
    # Determine color
    if status == "Initializing":
        color = (128, 128, 128)  # Gray
    elif status == "VIOLENCE":
        color = (0, 0, 255)  # Red
    else:
        color = (0, 255, 0)  # Green
    
    # Draw background box
    cv2.rectangle(frame, (10, 10), (500, 120), color, -1)
    
    # Draw status text
    cv2.putText(frame, status, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Draw confidence if available
    if confidence is not None:
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Collecting data...", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw segment range if applicable
    if segment_start is not None and segment_end is not None:
        time_start = segment_start / fps
        time_end = segment_end / fps
        cv2.putText(frame, f"Segment: {time_start:.1f}s - {time_end:.1f}s", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw frame info
    time_current = frame_idx / fps
    cv2.putText(frame, f"Frame: {frame_idx} ({time_current:.2f}s)", 
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def inference_video_mil(x3d_config_path, x3d_checkpoint_path, 
                       adapter_checkpoint_path, video_path, 
                       output_dir='outputs_mil', output_scale=1.0):
    """
    Run MIL-based inference on video.
    
    Pipeline:
    1. Extract 432-dim features using X3D with sliding window
    2. Apply MIL adapter on sliding window of 5 features
    3. Generate annotated video with predictions
    
    Args:
        x3d_config_path: Path to X3D config
        x3d_checkpoint_path: Path to X3D checkpoint
        adapter_checkpoint_path: Path to adapter checkpoint
        video_path: Path to input video
        output_dir: Output directory
        output_scale: Output video scale factor
    """
    print("="*80)
    print("MIL-based Violence Detection Inference")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load X3D config
    x3d_config = load_config(x3d_config_path)
    num_frames = x3d_config.get('num_frames', 16)
    frame_step = x3d_config.get('frame_step', 2)
    img_size = x3d_config.get('img_size', 224)
    inference_step = 15  # Fixed as per spec
    
    print(f"\nX3D Configuration:")
    print(f"  Frames per segment: {num_frames}")
    print(f"  Frame step: {frame_step}")
    print(f"  Inference step: {inference_step}")
    print(f"  Image size: {img_size}")
    
    # Load models
    x3d_model = load_x3d_model(x3d_config_path, x3d_checkpoint_path, device)
    adapter, adapter_type = load_adapter(adapter_checkpoint_path, input_dim=432, device=device)
    
    print(f"\nAdapter Configuration:")
    print(f"  Type: {adapter_type.upper()}")
    print(f"  Window size: 5")
    print(f"  Input dimension: 432")
    
    # Extract X3D features
    features, segment_start_frames = extract_x3d_features_from_video(
        video_path, x3d_model, device,
        num_frames=num_frames,
        frame_step=frame_step,
        inference_step=inference_step,
        img_size=img_size
    )
    
    if len(features) == 0:
        print("❌ Error: No features extracted from video")
        return
    
    # Run adapter predictions
    predictions, prediction_indices = predict_with_sliding_window(
        features, adapter, device, window_size=5
    )
    
    # Generate output video
    print(f"\n{'='*60}")
    print("Generating annotated video")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_mil_{adapter_type}.mp4")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate output dimensions
    out_width = int(orig_width * output_scale)
    out_height = int(orig_height * output_scale)
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)
    
    print(f"  Input resolution: {orig_width}x{orig_height}")
    print(f"  Output resolution: {out_width}x{out_height}")
    print(f"  FPS: {fps:.2f}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter")
    
    # Build prediction lookup
    # Map each frame to its status
    segment_span = (num_frames - 1) * frame_step + 1
    
    frame_status = {}  # frame_idx -> (status, confidence, segment_start, segment_end)
    
    # Initialize all frames as "Initializing"
    for frame_idx in range(total_frames):
        frame_status[frame_idx] = ("Initializing", None, None, None)
    
    # Apply predictions
    for pred_idx, (is_violence, confidence) in zip(prediction_indices, predictions):
        # This prediction corresponds to feature at pred_idx
        # Feature at pred_idx corresponds to segment starting at segment_start_frames[pred_idx]
        
        if pred_idx >= len(segment_start_frames):
            continue
        
        segment_start = segment_start_frames[pred_idx]
        segment_end = segment_start + segment_span
        
        status = "VIOLENCE" if is_violence else "NON-VIOLENCE"
        
        # Apply status to all frames in this segment
        for frame_idx in range(segment_start, min(segment_end, total_frames)):
            frame_status[frame_idx] = (status, confidence, segment_start, segment_end)
    
    # Read and annotate frames
    print(f"\n  Writing annotated video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame_idx in tqdm(range(total_frames), desc="  Progress"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get status for this frame
        status, confidence, seg_start, seg_end = frame_status.get(
            frame_idx, ("Initializing", None, None, None)
        )
        
        # Draw status
        frame = draw_status_on_frame(
            frame, status, confidence, frame_idx, fps,
            seg_start, seg_end
        )
        
        # Resize if needed
        if output_scale != 1.0:
            frame = cv2.resize(frame, (out_width, out_height),
                             interpolation=cv2.INTER_LINEAR)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Generate summary
    summary_path = os.path.join(output_dir, f"{video_name}_mil_{adapter_type}_summary.txt")
    
    violence_count = sum(1 for p, _ in predictions if p)
    violence_ratio = violence_count / len(predictions) if predictions else 0
    
    with open(summary_path, 'w') as f:
        f.write("MIL-based Violence Detection Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Video: {video_path}\n")
        f.write(f"Duration: {total_frames/fps:.2f}s ({total_frames} frames)\n\n")
        
        f.write(f"X3D Configuration:\n")
        f.write(f"  Model: X3D\n")
        f.write(f"  Checkpoint: {x3d_checkpoint_path}\n")
        f.write(f"  Frames per segment: {num_frames}\n")
        f.write(f"  Frame step: {frame_step}\n")
        f.write(f"  Segment span: {segment_span} frames\n")
        f.write(f"  Inference step: {inference_step}\n")
        f.write(f"  Feature dimension: 432\n\n")
        
        f.write(f"Adapter Configuration:\n")
        f.write(f"  Type: {adapter_type.upper()}\n")
        f.write(f"  Checkpoint: {adapter_checkpoint_path}\n")
        f.write(f"  Window size: 5\n")
        f.write(f"  Input dimension: 432\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Total features extracted: {len(features)}\n")
        f.write(f"  Total predictions made: {len(predictions)}\n")
        f.write(f"  Violence predictions: {violence_count}\n")
        f.write(f"  Non-violence predictions: {len(predictions) - violence_count}\n")
        f.write(f"  Violence ratio: {violence_ratio:.2%}\n\n")
        
        f.write("Detailed Predictions:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Pred':<6} {'Feature':<10} {'Frames':<20} {'Time':<20} {'Result':<15} {'Conf':<8}\n")
        f.write("-"*80 + "\n")
        
        for i, (pred_idx, (is_violence, confidence)) in enumerate(zip(prediction_indices, predictions)):
            if pred_idx >= len(segment_start_frames):
                continue
            
            seg_start = segment_start_frames[pred_idx]
            seg_end = seg_start + segment_span
            time_start = seg_start / fps
            time_end = seg_end / fps
            result = "VIOLENCE" if is_violence else "NON-VIOLENCE"
            
            f.write(f"{i+1:<6} {pred_idx:<10} {seg_start}-{seg_end:<15} "
                   f"{time_start:.2f}-{time_end:.2f}s{' ':<8} {result:<15} {confidence:.2%}\n")
    
    print(f"\n{'='*80}")
    print("✅ Inference completed successfully!")
    print(f"{'='*80}")
    print(f"Output video: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"\nResults:")
    print(f"  Violence: {violence_count}/{len(predictions)} ({violence_ratio:.2%})")
    print(f"  Non-violence: {len(predictions) - violence_count}/{len(predictions)} "
          f"({1-violence_ratio:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description='MIL-based Violence Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using LSTM adapter
  python infer_mil.py \\
    --x3d-config config/X3D/rfw.yaml \\
    --x3d-checkpoint checkpoints/x3d_rfw_violence/best_model_acc.pth \\
    --adapter checkpoints/mil_adapter/lstm_20260203_165436/best_model_acc.pth \\
    --video test_video.mp4
  
  # Using MLP adapter at half resolution
  python infer_mil.py \\
    --x3d-config config/X3D/rfw.yaml \\
    --x3d-checkpoint checkpoints/x3d_rfw_violence/best_model_acc.pth \\
    --adapter checkpoints/mil_adapter/mlp_20260203_165324/best_model_acc.pth \\
    --video test_video.mp4 \\
    --scale 0.5
  
  # Process folder
  python infer_mil.py \\
    --x3d-config config/X3D/rfw.yaml \\
    --x3d-checkpoint checkpoints/x3d_rfw_violence/best_model_acc.pth \\
    --adapter checkpoints/mil_adapter/conv1d_20260203_165556/best_model_acc.pth \\
    --video-folder ./test_videos
        """
    )
    
    parser.add_argument('--x3d-config', type=str, required=True,
                       help='Path to X3D config YAML')
    parser.add_argument('--x3d-checkpoint', type=str, required=True,
                       help='Path to X3D checkpoint (.pth)')
    parser.add_argument('--adapter', type=str, required=True,
                       help='Path to MIL adapter checkpoint (.pth)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video')
    parser.add_argument('--video-folder', type=str, default=None,
                       help='Path to folder with videos')
    parser.add_argument('--output', type=str, default='outputs/outputs_mil',
                       help='Output directory (default: outputs_mil)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Output video scale (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.x3d_config):
        raise FileNotFoundError(f"X3D config not found: {args.x3d_config}")
    
    if not os.path.exists(args.x3d_checkpoint):
        raise FileNotFoundError(f"X3D checkpoint not found: {args.x3d_checkpoint}")
    
    if not os.path.exists(args.adapter):
        raise FileNotFoundError(f"Adapter checkpoint not found: {args.adapter}")
    
    if args.video is None and args.video_folder is None:
        raise ValueError("Either --video or --video-folder must be provided")
    
    if args.video and args.video_folder:
        raise ValueError("Cannot specify both --video and --video-folder")
    
    # Process single video
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        
        inference_video_mil(
            args.x3d_config,
            args.x3d_checkpoint,
            args.adapter,
            args.video,
            args.output,
            args.scale
        )
    
    # Process folder
    else:
        if not os.path.isdir(args.video_folder):
            raise FileNotFoundError(f"Folder not found: {args.video_folder}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(args.video_folder).glob(f'*{ext}'))
        
        video_files = sorted(list(set(video_files)))
        
        if not video_files:
            print(f"❌ No videos found in {args.video_folder}")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(video_files)} video(s)")
        print(f"{'='*80}\n")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
            try:
                inference_video_mil(
                    args.x3d_config,
                    args.x3d_checkpoint,
                    args.adapter,
                    str(video_path),
                    args.output,
                    args.scale
                )
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print(f"✅ Batch processing completed!")
        print(f"Results: {args.output}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

# uv run  python main.py     --x3d-config config/X3D/rfw.yaml     --x3d-checkpoint checkpoints/x3d_rfw_violence/best_model_acc.pth     --adapter checkpoints/mil_adapter/conv1d_20260203_165556/best_model_acc.pth     --video-folder /home/atin-ct3/action_recognition/data/Test-Data