import os
import yaml
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import get_model


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_memory_usage(model, device):
    """
    Calculate model memory usage.
    
    Args:
        model: PyTorch model
        device: torch.device
        
    Returns:
        tuple: (params_memory_mb, total_memory_mb)
    """
    # Calculate parameter memory
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    params_memory_mb = (param_size + buffer_size) / 1024 / 1024
    
    # Get GPU memory if available
    if device.type == 'cuda':
        torch.cuda.synchronize()
        allocated_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved_memory_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
        return params_memory_mb, allocated_memory_mb, reserved_memory_mb
    else:
        return params_memory_mb, params_memory_mb, params_memory_mb


def load_model(config, checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        config (dict): Model configuration
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model_name = config.get('model_name', 'TimeSformer')
    
    if model_name == 'TimeSformer':
        model = get_model(**config)
    elif model_name == 'LSTM_CNN':
        model, _, _ = get_model(**config)
    else:
        model = get_model(**config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model


def extract_frames_with_step(video_path, num_frames=8, inference_step=4, 
                             frame_step=1, target_size=224):
    """
    Extract frames from video with sliding window and temporal subsampling.
    
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames per sequence
        inference_step (int): Step size between consecutive predictions (sliding window)
        frame_step (int): Step between frames within each sequence (temporal subsampling)
                         (1=consecutive frames, 2=skip 1 frame, 4=skip 3 frames, etc.)
        target_size (int): Target frame size
        
    Returns:
        tuple: (sequences, frame_indices, all_frames, fps)
            - sequences: List of frame sequences, each of shape (num_frames, H, W, 3)
            - frame_indices: List of starting frame indices for each sequence
            - all_frames: All frames from video (resized)
            - fps: Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {total_frames/fps:.2f}s duration")
    
    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (target_size, target_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        all_frames.append(frame)
    
    cap.release()
    
    # Calculate required span for sequence with frame_step
    required_span = (num_frames - 1) * frame_step + 1
    
    print(f"Frame extraction config:")
    print(f"  Frames per sequence: {num_frames}")
    print(f"  Frame step (temporal subsampling): {frame_step}")
    print(f"  Required span per sequence: {required_span} frames (~{required_span/fps:.2f}s)")
    print(f"  Inference step (sliding window): {inference_step}")
    
    # Extract sequences with sliding window and frame stepping
    sequences = []
    frame_indices = []
    
    # Slide the window by inference_step
    for window_start in range(0, len(all_frames) - required_span + 1, inference_step):
        # Extract frames with frame_step within this window
        frame_indices_in_seq = [window_start + i * frame_step for i in range(num_frames)]
        
        # Get frames for this sequence
        sequence = [all_frames[idx] for idx in frame_indices_in_seq]
        
        sequences.append(np.array(sequence))
        frame_indices.append(window_start)
    
    print(f"\nExtracted {len(sequences)} sequences:")
    print(f"  Covering {len(all_frames)} frames")
    print(f"  Time range: {0:.2f}s to {len(all_frames)/fps:.2f}s")
    
    return sequences, frame_indices, all_frames, fps


def preprocess_sequence(sequence, model_name='TimeSformer'):
    """
    Preprocess a sequence for model input.
    
    Args:
        sequence (np.ndarray): Frame sequence of shape (num_frames, H, W, 3)
        model_name (str): Model name for specific preprocessing
        
    Returns:
        torch.Tensor: Preprocessed sequence of shape (1, num_frames, 3, H, W)
    """
    from torchvision import transforms
    
    # Standard normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Process each frame
    processed_frames = []
    for frame in sequence:
        # Convert to PIL Image
        pil_frame = transforms.ToPILImage()(frame)
        tensor_frame = transform(pil_frame)  # (3, H, W)
        processed_frames.append(tensor_frame)
    
    # Stack frames: (num_frames, 3, H, W)
    sequence_tensor = torch.stack(processed_frames)
    
    # Add batch dimension: (1, num_frames, 3, H, W)
    sequence_tensor = sequence_tensor.unsqueeze(0)
    
    return sequence_tensor


def predict_violence(model, sequence_tensor, device):
    """
    Predict violence for a sequence.
    
    Args:
        model (torch.nn.Module): Trained model
        sequence_tensor (torch.Tensor): Preprocessed sequence
        device (torch.device): Device
        
    Returns:
        tuple: (is_violence, confidence)
            - is_violence (bool): True if violence detected
            - confidence (float): Prediction confidence [0-1]
    """
    with torch.no_grad():
        sequence_tensor = sequence_tensor.to(device)
        outputs = model(sequence_tensor)  # (1, num_classes)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        is_violence = predicted.item() == 1
        confidence_val = confidence.item()
    
    return is_violence, confidence_val


def draw_prediction_on_frame(frame, is_violence, confidence, frame_idx):
    """
    Draw prediction on frame.
    
    Args:
        frame (np.ndarray): Frame in BGR format
        is_violence (bool): Violence prediction
        confidence (float): Prediction confidence
        frame_idx (int): Frame index
        
    Returns:
        np.ndarray: Frame with annotations
    """
    frame = frame.copy()
    
    # Set color based on prediction
    if is_violence:
        color = (0, 0, 255)  # Red for violence
        label = "VIOLENCE"
    else:
        color = (0, 255, 0)  # Green for non-violence
        label = "NON-VIOLENCE"
    
    # Draw rectangle at top
    cv2.rectangle(frame, (10, 10), (400, 80), color, -1)
    
    # Draw text
    cv2.putText(frame, label, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw frame number
    cv2.putText(frame, f"Frame: {frame_idx}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def inference_video(config_path, checkpoint_path, video_path, output_dir='outputs'):
    """
    Run inference on video and create annotated output.
    
    Configuration parameters:
        num_frames: Number of frames in each model input sequence
        frame_step: Step between frames WITHIN each sequence (temporal subsampling)
        inference_step: Step between consecutive predictions (sliding window overlap)
    """
    # Load config
    config = load_config(config_path)
    
    # Extract inference parameters
    num_frames = config.get('num_frames', 8)
    frame_step = config.get('frame_step', 1)  # Step within sequence
    inference_step = config.get('inference_step', 4)  # Step between predictions
    img_size = config.get('img_size', 224)
    model_name = config.get('model_name', 'TimeSformer')
    
    print(f"\n{'='*60}")
    print(f"Inference Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"\nFrame Processing:")
    print(f"  Frames per sequence: {num_frames}")
    print(f"  Frame step (within sequence): {frame_step}")
    print(f"    - Temporal span per sequence: {(num_frames-1)*frame_step + 1} frames")
    print(f"  Inference step (sliding window): {inference_step}")
    print(f"    - Overlap between predictions: {num_frames - inference_step} frames")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Setup device
    device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    
    # Calculate model parameters and memory
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_mem, allocated_mem, reserved_mem = get_model_memory_usage(model, device)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {params_mem:.2f} MB")
    if device.type == 'cuda':
        print(f"  VRAM allocated: {allocated_mem:.2f} MB")
        print(f"  VRAM reserved: {reserved_mem:.2f} MB")
    
    # Extract frames with sliding window and frame stepping
    print(f"\nProcessing video: {video_path}")
    sequences, sequence_start_indices, all_frames, fps = extract_frames_with_step(
        video_path, 
        num_frames=num_frames, 
        inference_step=inference_step,
        frame_step=frame_step,
        target_size=img_size
    )
    
    if len(sequences) == 0:
        print(f"❌ Error: No sequences could be extracted from video")
        return
    
    # Run predictions for each sequence
    predictions = []
    
    model.eval()
    
    # Measure peak VRAM during inference
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    print(f"\nRunning inference on {len(sequences)} sequences...")
    for i, sequence in enumerate(tqdm(sequences)):
        # Preprocess
        sequence_tensor = preprocess_sequence(sequence, model_name)
        
        # Predict
        is_violence, confidence = predict_violence(model, sequence_tensor, device)
        
        predictions.append({
            'start_frame': sequence_start_indices[i],
            'end_frame': sequence_start_indices[i] + (num_frames - 1) * frame_step + 1,
            'is_violence': is_violence,
            'confidence': confidence
        })
    
    # Get peak VRAM usage
    peak_vram_mb = 0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"\nPeak VRAM during inference: {peak_vram_mb:.2f} MB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output video
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    
    print(f"\nGenerating annotated video: {output_path}")
    
    # Get original video dimensions
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
    
    # Annotate each frame
    for frame_idx in tqdm(range(len(all_frames)), desc="Writing video"):
        # Find prediction for current frame
        current_prediction = None
        for pred in predictions:
            if pred['start_frame'] <= frame_idx < pred['end_frame']:
                current_prediction = pred
                break
        
        # Get original frame (need to read again for original size)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            break
        
        # Draw prediction if available
        if current_prediction is not None:
            frame = draw_prediction_on_frame(
                frame,
                current_prediction['is_violence'],
                current_prediction['confidence'],
                frame_idx
            )
        
        out.write(frame)
    
    out.release()
    
    # Generate summary report
    summary_path = os.path.join(output_dir, f"{video_name}_summary.txt")
    
    violence_count = sum(1 for p in predictions if p['is_violence'])
    violence_ratio = violence_count / len(predictions) if predictions else 0
    
    with open(summary_path, 'w') as f:
        f.write(f"Violence Detection Inference Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        
        f.write(f"Model Information:\n")
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Trainable parameters: {trainable_params:,}\n")
        f.write(f"  Model size: {params_mem:.2f} MB\n")
        if device.type == 'cuda':
            f.write(f"  VRAM allocated: {allocated_mem:.2f} MB\n")
            f.write(f"  VRAM reserved: {reserved_mem:.2f} MB\n")
            f.write(f"  Peak VRAM during inference: {peak_vram_mb:.2f} MB\n")
        f.write(f"\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Frames per sequence: {num_frames}\n")
        f.write(f"  Frame step (within sequence): {frame_step}\n")
        f.write(f"  Inference step (sliding window): {inference_step}\n")
        f.write(f"  Sequence temporal span: {(num_frames-1)*frame_step + 1} frames (~{((num_frames-1)*frame_step + 1)/fps:.2f}s)\n")
        f.write(f"  Prediction overlap: {num_frames - inference_step} frames (~{(num_frames - inference_step)/fps:.2f}s)\n\n")
        
        f.write(f"Video Info:\n")
        f.write(f"  Total frames: {len(all_frames)}\n")
        f.write(f"  Duration: {len(all_frames)/fps:.2f}s\n")
        f.write(f"  FPS: {fps:.2f}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Total sequences predicted: {len(predictions)}\n")
        f.write(f"  Violence sequences: {violence_count}\n")
        f.write(f"  Non-violence sequences: {len(predictions) - violence_count}\n")
        f.write(f"  Violence ratio: {violence_ratio:.2%}\n\n")
        
        f.write(f"Detailed Predictions:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Seq':<5} {'Frames':<20} {'Time':<20} {'Result':<15} {'Confidence':<12}\n")
        f.write(f"{'-'*80}\n")
        
        for i, pred in enumerate(predictions):
            time_start = pred['start_frame'] / fps
            time_end = pred['end_frame'] / fps
            result = "VIOLENCE" if pred['is_violence'] else "NON-VIOLENCE"
            f.write(f"{i+1:<5} {pred['start_frame']}-{pred['end_frame']:<15} "
                   f"{time_start:.2f}-{time_end:.2f}s{' ':<10} {result:<15} {pred['confidence']:.2%}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Inference completed successfully!")
    print(f"{'='*60}")
    print(f"Output video: {output_path}")
    print(f"Summary report: {summary_path}")
    print(f"\nModel Resources:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {params_mem:.2f} MB")
    if device.type == 'cuda':
        print(f"  Peak VRAM: {peak_vram_mb:.2f} MB")
    print(f"\nResults:")
    print(f"  Violence sequences: {violence_count}/{len(predictions)} ({violence_ratio:.2%})")
    print(f"  Non-violence sequences: {len(predictions) - violence_count}/{len(predictions)} ({1-violence_ratio:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description='Video Violence Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/TimeSformer/rfw.yaml --checkpoint checkpoints/best.pth --video test.mp4
  python main.py --config config/TimeSformer/rfw.yaml --checkpoint checkpoints/best.pth --video test.mp4 --output results/
        """
    )
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Run inference
    inference_video(args.config, args.checkpoint, args.video, args.output)


if __name__ == "__main__":
    main()