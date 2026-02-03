"""
Utility functions for model analysis, visualization, and performance mining.
Includes attention visualization, failure analysis, and prediction debugging.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import get_model
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_for_analysis(config: dict, checkpoint_path: str, device: torch.device):
    """
    Load model for analysis with gradient tracking enabled.
    Uses same logic as main.py load_model().
    
    Args:
        config: Model configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Torch device
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model - same logic as main.py
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


def extract_sequence_from_video(
    video_path: str,
    start_time: Optional[float] = None,
    num_frames: int = 8,
    frame_step: int = 1,
    target_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Extract a single sequence from video at specified or random time.
    Uses same frame extraction logic as main.py extract_frames_with_step().
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds (None for random)
        num_frames: Number of frames to extract
        frame_step: Step between frames within each sequence (temporal subsampling)
        target_size: Target frame size
        
    Returns:
        Tuple of (sequence, original_frames, fps, start_frame_idx)
            - sequence: Resized frames (num_frames, H, W, 3) RGB
            - original_frames: Original resolution frames (num_frames, H, W, 3) BGR
            - fps: Video FPS
            - start_frame_idx: Starting frame index
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate required span - same logic as main.py
    required_span = (num_frames - 1) * frame_step + 1
    max_start_frame = total_frames - required_span
    
    if max_start_frame < 0:
        raise ValueError(f"Video too short. Needs {required_span} frames, has {total_frames}")
    
    # Determine start frame
    if start_time is not None:
        start_frame_idx = int(start_time * fps)
        if start_frame_idx > max_start_frame:
            print(f"Warning: start_time {start_time}s exceeds valid range. Using max valid position.")
            start_frame_idx = max_start_frame
        if start_frame_idx < 0:
            start_frame_idx = 0
    else:
        start_frame_idx = random.randint(0, max_start_frame)
    
    # Extract frames - same logic as main.py
    sequence_resized = []
    original_frames = []
    
    for i in range(num_frames):
        frame_idx = start_frame_idx + i * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            raise IOError(f"Failed to read frame {frame_idx}")
        
        # Store original (BGR)
        original_frames.append(frame.copy())
        
        # Convert BGR to RGB - same as main.py
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize - same as main.py
        frame_resized = cv2.resize(frame_rgb, (target_size, target_size), 
                                   interpolation=cv2.INTER_LINEAR)
        sequence_resized.append(frame_resized)
    
    cap.release()
    
    return (np.array(sequence_resized), np.array(original_frames), 
            fps, start_frame_idx)


def preprocess_sequence(sequence: np.ndarray, model_name: str = 'TimeSformer') -> torch.Tensor:
    """
    Preprocess a sequence for model input.
    EXACT same logic as main.py preprocess_sequence().
    
    Args:
        sequence (np.ndarray): Frame sequence of shape (num_frames, H, W, 3)
        model_name (str): Model name for specific preprocessing
        
    Returns:
        torch.Tensor: Preprocessed sequence
            - TimeSformer: shape (1, num_frames, 3, H, W)
            - I3D/X3D: shape (1, 3, num_frames, H, W)
    """
    from torchvision import transforms
    
    # Standard normalization - same as main.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Process each frame - same as main.py
    processed_frames = []
    for frame in sequence:
        # Convert to PIL Image
        pil_frame = transforms.ToPILImage()(frame)
        tensor_frame = transform(pil_frame)  # (3, H, W)
        processed_frames.append(tensor_frame)
    
    # Stack frames based on model type - EXACT same logic as main.py
    if model_name in ['I3D', 'X3D']:
        # I3D/X3D format: (1, 3, T, H, W)
        # Stack to (T, 3, H, W) then rearrange to (3, T, H, W)
        sequence_tensor = torch.stack(processed_frames)  # (T, 3, H, W)
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3)  # (3, T, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, 3, T, H, W)
    else:
        # TimeSformer and others: (1, T, 3, H, W)
        sequence_tensor = torch.stack(processed_frames)  # (T, 3, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, T, 3, H, W)
    
    return sequence_tensor


def predict_violence(model, sequence_tensor: torch.Tensor, device: torch.device) -> Tuple[bool, float]:
    """
    Predict violence for a sequence.
    EXACT same logic as main.py predict_violence().
    
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


def get_attention_maps_timesformer(
    model,
    input_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Extract attention maps from TimeSformer model.
    
    Args:
        model: TimeSformer model
        input_tensor: Input tensor (1, T, 3, H, W)
        device: Torch device
        
    Returns:
        Tuple of (output logits, list of attention maps per layer)
    """
    attention_maps = []
    
    def hook_fn(module, input, output):
        # TimeSformer attention output is (batch, num_heads, seq_len, seq_len)
        if hasattr(output, 'shape') and len(output.shape) == 4:
            attention_maps.append(output.detach().cpu())
    
    hooks = []
    
    # Access the underlying model
    if hasattr(model, 'model'):
        timesformer = model.model
    else:
        timesformer = model
    
    # Register hooks on attention layers
    if hasattr(timesformer, 'encoder') and hasattr(timesformer.encoder, 'layer'):
        for layer in timesformer.encoder.layer:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                hook = layer.attention.self.register_forward_hook(hook_fn)
                hooks.append(hook)
    
    # Forward pass
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return output, attention_maps


def get_gradcam_attention(
    model,
    input_tensor: torch.Tensor,
    target_class: Optional[int],
    device: torch.device,
    model_name: str = 'TimeSformer'
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """
    Compute attention visualization for different model types.
    - TimeSformer: Extract CLS token attention weights from last layer
    - I3D/X3D: Standard GradCAM on convolutional feature maps
    
    Args:
        model: Model to analyze
        input_tensor: Input tensor
        target_class: Target class (None to use predicted class)
        device: Torch device
        model_name: Model name
        
    Returns:
        Tuple of (output logits, attention heatmaps per frame, predicted class)
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Get number of frames based on model type - consistent with main.py
    if model_name in ['I3D', 'X3D']:
        num_frames = input_tensor.shape[2]  # (1, 3, T, H, W)
    else:
        num_frames = input_tensor.shape[1]  # (1, T, 3, H, W)
    
    img_size = 224  # Default image size
    
    # =========================================================================
    # TimeSformer: Use CLS token attention weights (NOT GradCAM)
    # =========================================================================
    if model_name == 'TimeSformer':
        return _get_timesformer_attention(model, input_tensor, target_class, device, num_frames, img_size)
    
    # =========================================================================
    # I3D/X3D: Use standard GradCAM on CNN feature maps
    # =========================================================================
    else:
        return _get_cnn_gradcam_attention(model, input_tensor, target_class, device, model_name, num_frames, img_size)


def _get_timesformer_attention(
    model,
    input_tensor: torch.Tensor,
    target_class: Optional[int],
    device: torch.device,
    num_frames: int,
    img_size: int = 224
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """
    Extract CLS token attention weights from TimeSformer.
    
    TimeSformer uses patches (tokens). To visualize attention:
    1. Get attention weights from the last transformer block
    2. Extract what the CLS token attends to (row 0 of attention matrix)
    3. Reshape the 1D token sequence back to (T, H, W) spatial grid
    4. Upscale to original image size
    """
    # Storage for attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        """
        Hook to capture attention weights.
        Different implementations return attention differently:
        - Some return (attn_output, attn_weights)
        - Some just return attn_output and store weights internally
        """
        # Try to get attention weights from the module
        if hasattr(module, 'attn_weights'):
            attention_weights.append(module.attn_weights.detach().cpu())
        elif isinstance(output, tuple) and len(output) >= 2:
            # (attn_output, attn_weights) format
            if output[1] is not None:
                attention_weights.append(output[1].detach().cpu())
    
    hooks = []
    
    # Access the underlying TimeSformer model
    if hasattr(model, 'model'):
        base = model.model
    else:
        base = model
    
    # Try to find and hook attention layers
    # Path varies by implementation (HuggingFace vs timm vs custom)
    target_attention = None
    
    # HuggingFace TimesformerModel
    if hasattr(base, 'timesformer') and hasattr(base.timesformer, 'encoder'):
        encoder = base.timesformer.encoder
        if hasattr(encoder, 'layer'):
            # Hook the last layer's attention
            last_layer = encoder.layer[-1]
            if hasattr(last_layer, 'attention'):
                target_attention = last_layer.attention
    
    # Alternative: Direct encoder access
    elif hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        last_layer = base.encoder.layer[-1]
        if hasattr(last_layer, 'attention'):
            target_attention = last_layer.attention
    
    # timm-style ViT/TimeSformer
    elif hasattr(base, 'blocks'):
        last_block = base.blocks[-1]
        if hasattr(last_block, 'attn'):
            target_attention = last_block.attn
    
    if target_attention is not None:
        hook = target_attention.register_forward_hook(attention_hook)
        hooks.append(hook)
    
    # Forward pass - try with output_attentions if available
    with torch.no_grad():
        try:
            # HuggingFace style: request attention output
            output = model(input_tensor, output_attentions=True)
            
            # Check if output contains attentions
            if hasattr(output, 'attentions') and output.attentions is not None:
                # Use returned attentions (list of attention per layer)
                last_attn = output.attentions[-1]  # (B, Heads, SeqLen, SeqLen)
                attention_weights = [last_attn.cpu()]
            
            # Get logits
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
                
        except TypeError:
            # Model doesn't support output_attentions, use hook result
            output = model(input_tensor)
            logits = output
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get prediction
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()
    
    # Process attention weights
    if len(attention_weights) > 0:
        attn = attention_weights[-1]  # Use last layer
        
        # Shape: (Batch, Heads, SeqLen, SeqLen)
        # Average over heads
        if len(attn.shape) == 4:
            attn = attn.mean(dim=1)  # (Batch, SeqLen, SeqLen)
        
        # Get CLS token attention (row 0 attending to all patches)
        # Exclude CLS token itself (index 0)
        cls_attn = attn[0, 0, 1:]  # (SeqLen-1,) = (num_patches,)
        
        # Reshape patches back to spatial grid
        # TimeSformer patch layout: T * (H/patch_size) * (W/patch_size)
        num_patches = cls_attn.shape[0]
        
        # Assume 16x16 patches on 224x224 image = 14x14 spatial grid per frame
        patch_size = 16
        spatial_size = img_size // patch_size  # 14
        patches_per_frame = spatial_size * spatial_size  # 196
        
        # Check if we have temporal * spatial patches
        if num_patches == num_frames * patches_per_frame:
            # Reshape to (T, H_patches, W_patches)
            attn_map = cls_attn.numpy().reshape(num_frames, spatial_size, spatial_size)
        elif num_patches == patches_per_frame:
            # Only spatial patches (replicate for all frames)
            attn_map = cls_attn.numpy().reshape(spatial_size, spatial_size)
            attn_map = np.stack([attn_map] * num_frames, axis=0)
        else:
            # Unknown layout - try best guess
            print(f"Warning: Unexpected patch count {num_patches}. Attempting reshape...")
            total_spatial = num_patches // num_frames if num_patches >= num_frames else num_patches
            side = int(np.sqrt(total_spatial))
            if side * side * num_frames == num_patches:
                attn_map = cls_attn.numpy().reshape(num_frames, side, side)
            else:
                # Fallback: uniform attention
                print(f"Could not reshape {num_patches} patches. Using uniform.")
                attn_map = np.ones((num_frames, spatial_size, spatial_size)) * 0.5
        
        # Normalize and upscale each frame
        attention_maps = []
        for t in range(num_frames):
            frame_attn = attn_map[t]
            
            # Normalize to [0, 1]
            frame_attn = frame_attn - frame_attn.min()
            if frame_attn.max() > 0:
                frame_attn = frame_attn / frame_attn.max()
            
            # Upscale to original image size
            frame_attn = cv2.resize(frame_attn.astype(np.float32), (img_size, img_size), 
                                    interpolation=cv2.INTER_LINEAR)
            attention_maps.append(frame_attn)
        
        attention_maps = np.array(attention_maps)
    else:
        # Fallback: try to extract from model internals or use uniform
        print("Warning: Could not extract TimeSformer attention weights. Using uniform fallback.")
        attention_maps = np.ones((num_frames, img_size, img_size)) * 0.5
    
    return logits, attention_maps, pred_class


def _get_cnn_gradcam_attention(
    model,
    input_tensor: torch.Tensor,
    target_class: Optional[int],
    device: torch.device,
    model_name: str,
    num_frames: int,
    img_size: int = 224
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """
    Standard GradCAM for CNN-based models (I3D, X3D).
    Works on 3D feature maps of shape (B, C, T, H, W).
    """
    input_tensor.requires_grad = True
    
    # Storage for activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach())
        else:
            if grad_output is not None:
                gradients.append(grad_output.detach())
    
    # Find target layer
    target_layer = None
    
    if hasattr(model, 'model'):
        base = model.model
    else:
        base = model
    
    # For X3D/I3D, target the last conv layer
    if hasattr(base, 'blocks'):
        target_layer = base.blocks[-2]  # Second to last block
    
    if target_layer is None:
        print(f"Warning: Could not find target layer for {model_name}. Using fallback.")
        with torch.no_grad():
            output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        attention_maps = np.ones((num_frames, img_size, img_size)) * 0.5
        return output, attention_maps, pred_class
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = pred_class
    
    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Compute GradCAM
    if len(activations) > 0 and len(gradients) > 0:
        activation = activations[0]
        gradient = gradients[0]
        
        # Shape: (B, C, T, H, W)
        # Global average pooling of gradients over spatial dims
        weights = gradient.mean(dim=(3, 4), keepdim=True)  # (B, C, T, 1, 1)
        cam = (weights * activation).sum(dim=1)  # (B, T, H, W)
        cam = cam.squeeze(0).cpu().numpy()  # (T, H, W)
        
        # Normalize and resize each frame
        attention_maps = []
        for i in range(min(len(cam), num_frames)):
            frame_cam = cam[i]
            
            # ReLU (only positive contributions)
            frame_cam = np.maximum(frame_cam, 0)
            
            # Normalize to [0, 1]
            frame_cam = frame_cam - frame_cam.min()
            if frame_cam.max() > 0:
                frame_cam = frame_cam / frame_cam.max()
            
            # Resize to original image size
            frame_cam = cv2.resize(frame_cam, (img_size, img_size), 
                                   interpolation=cv2.INTER_LINEAR)
            attention_maps.append(frame_cam)
        
        # Pad if needed
        while len(attention_maps) < num_frames:
            attention_maps.append(np.ones((img_size, img_size)) * 0.5)
        
        attention_maps = np.array(attention_maps[:num_frames])
    else:
        attention_maps = np.ones((num_frames, img_size, img_size)) * 0.5
    
    return output, attention_maps, pred_class

def visualize_prediction_with_attention(
    config_path: str,
    checkpoint_path: str,
    video_path: str,
    start_time: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10),
    class_names: List[str] = ['Non-Violence', 'Violence'],
    attention_cmap: str = 'Greys'
) -> Dict:
    """
    Main function to visualize model prediction with attention maps.
    Uses same loading and preprocessing logic as main.py.
    
    Args:
        config_path: Path to model config YAML
        checkpoint_path: Path to model checkpoint
        video_path: Path to input video
        start_time: Start time in seconds (None for random)
        save_path: Path to save figure (None for display only)
        figsize: Figure size
        class_names: List of class names
        attention_cmap: Colormap for attention ('Greys', 'Blues', 'Purples', 'Oranges', etc.)
        
    Returns:
        Dictionary with prediction results and metadata
    """
    # Load config - same as main.py
    config = load_config(config_path)
    
    # Setup device - same as main.py
    device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else 'cpu')
    
    # Extract inference parameters - same as main.py
    model_name = config.get('model_name', 'TimeSformer')
    num_frames = config.get('num_frames', 8)
    frame_step = config.get('frame_step', 1)
    img_size = config.get('img_size', 224)
    
    print(f"Loading model: {model_name}")
    model = load_model_for_analysis(config, checkpoint_path, device)
    
    # Extract sequence - using same logic as main.py
    print(f"Extracting sequence from video...")
    sequence, original_frames, fps, start_frame_idx = extract_sequence_from_video(
        video_path,
        start_time=start_time,
        num_frames=num_frames,
        frame_step=frame_step,
        target_size=img_size
    )
    
    start_time_actual = start_frame_idx / fps
    end_time = (start_frame_idx + (num_frames - 1) * frame_step) / fps
    
    print(f"Sequence: frames {start_frame_idx} to {start_frame_idx + (num_frames-1)*frame_step}")
    print(f"Time: {start_time_actual:.2f}s to {end_time:.2f}s")
    
    # Preprocess - EXACT same function as main.py
    input_tensor = preprocess_sequence(sequence, model_name)
    
    # Get prediction and attention
    print("Computing attention maps...")
    output, attention_maps, pred_class = get_gradcam_attention(
        model, input_tensor, None, device, model_name
    )
    
    # Get probabilities - same logic as main.py predict_violence
    probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_label = class_names[pred_class]
    confidence = probs[pred_class]
    
    print(f"Prediction: {pred_label} ({confidence:.2%})")
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, num_frames, figure=fig, height_ratios=[1.2, 1, 0.3])
    
    # Row 1: Original frames with prediction overlay
    for i in range(num_frames):
        ax = fig.add_subplot(gs[0, i])
        
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB)
        
        # Resize for display
        frame_display = cv2.resize(frame_rgb, (img_size, img_size))
        
        ax.imshow(frame_display)
        
        # Add frame info
        frame_idx = start_frame_idx + i * frame_step
        frame_time = frame_idx / fps
        ax.set_title(f'Frame {frame_idx}\n{frame_time:.2f}s', fontsize=9)
        ax.axis('off')
        
        # Add prediction box on first frame
        if i == 0:
            color = 'red' if pred_class == 1 else 'green'
            ax.text(0.5, 1.15, f'{pred_label}: {confidence:.1%}',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   ha='center', va='bottom', color=color,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Row 2: Attention heatmaps overlaid on frames (with monotone colormap)
    for i in range(num_frames):
        ax = fig.add_subplot(gs[1, i])
        
        # Get frame and attention
        frame_rgb = cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        
        # Normalize frame for overlay
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Get attention map
        attn = attention_maps[i]
        
        # Apply monotone colormap to attention
        attn_colored = plt.cm.get_cmap(attention_cmap)(attn)[:, :, :3]
        
        # Blend frame with attention
        alpha = 0.5
        blended = (1 - alpha) * frame_normalized + alpha * attn_colored
        blended = np.clip(blended, 0, 1)
        
        ax.imshow(blended)
        ax.set_title(f'Attention {i+1}', fontsize=9)
        ax.axis('off')
    
    # Row 3: Attention intensity bar
    ax_bar = fig.add_subplot(gs[2, :])
    
    # Compute mean attention per frame
    mean_attention = [attention_maps[i].mean() for i in range(num_frames)]
    
    # Use monotone colormap for bar colors
    cmap = plt.cm.get_cmap(attention_cmap)
    colors = cmap(np.array(mean_attention) / max(mean_attention) if max(mean_attention) > 0 else np.zeros(num_frames))
    bars = ax_bar.bar(range(num_frames), mean_attention, color=colors)
    ax_bar.set_xlabel('Frame Index')
    ax_bar.set_ylabel('Mean Attention')
    ax_bar.set_title('Attention Distribution Across Frames')
    ax_bar.set_xticks(range(num_frames))
    ax_bar.set_xticklabels([f'{start_frame_idx + i*frame_step}' for i in range(num_frames)])
    
    # Add colorbar with monotone colormap
    sm = plt.cm.ScalarMappable(cmap=attention_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Attention Intensity')
    
    # Overall title
    video_name = Path(video_path).name
    fig.suptitle(
        f'Video: {video_name}\n'
        f'Model: {model_name} | Time: {start_time_actual:.2f}s - {end_time:.2f}s | '
        f'Prediction: {pred_label} ({confidence:.1%})',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Return results
    return {
        'video_path': video_path,
        'start_time': start_time_actual,
        'end_time': end_time,
        'start_frame': start_frame_idx,
        'num_frames': num_frames,
        'frame_step': frame_step,
        'fps': fps,
        'prediction': pred_label,
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        'mean_attention_per_frame': mean_attention,
        'model_name': model_name
    }


def analyze_multiple_sequences(
    config_path: str,
    checkpoint_path: str,
    video_path: str,
    time_points: List[float],
    save_dir: Optional[str] = None,
    class_names: List[str] = ['Non-Violence', 'Violence'],
    attention_cmap: str = 'Greys'
) -> List[Dict]:
    """
    Analyze multiple time points in a video.
    
    Args:
        config_path: Path to config
        checkpoint_path: Path to checkpoint
        video_path: Path to video
        time_points: List of start times in seconds
        save_dir: Directory to save figures
        class_names: Class names
        attention_cmap: Colormap for attention visualization
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, start_time in enumerate(time_points):
        print(f"\n{'='*50}")
        print(f"Analyzing sequence {i+1}/{len(time_points)} at {start_time:.2f}s")
        print(f"{'='*50}")
        
        save_path = None
        if save_dir:
            video_name = Path(video_path).stem
            save_path = os.path.join(save_dir, f"{video_name}_t{start_time:.1f}s.png")
        
        try:
            result = visualize_prediction_with_attention(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                video_path=video_path,
                start_time=start_time,
                save_path=save_path,
                class_names=class_names,
                attention_cmap=attention_cmap
            )
            results.append(result)
        except Exception as e:
            print(f"Error at {start_time}s: {e}")
            continue
    
    return results


def generate_failure_analysis_report(
    results: List[Dict],
    ground_truth: Optional[List[int]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a text report analyzing predictions.
    
    Args:
        results: List of result dictionaries from visualize_prediction_with_attention
        ground_truth: Optional ground truth labels
        output_path: Optional path to save report
        
    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PREDICTION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    predictions = [r['predicted_class'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    lines.append(f"Total sequences analyzed: {len(results)}")
    lines.append(f"Class distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    lines.append(f"Average confidence: {np.mean(confidences):.2%}")
    lines.append(f"Min confidence: {np.min(confidences):.2%}")
    lines.append(f"Max confidence: {np.max(confidences):.2%}")
    lines.append("")
    
    if ground_truth:
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions)
        lines.append(f"Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
        
        # Find failures
        failures = [(i, results[i]) for i, (p, g) in enumerate(zip(predictions, ground_truth)) if p != g]
        lines.append(f"Failures: {len(failures)}")
        lines.append("")
        
        if failures:
            lines.append("-" * 40)
            lines.append("FAILURE ANALYSIS")
            lines.append("-" * 40)
            for idx, result in failures:
                lines.append(f"\nFailure at sequence {idx}:")
                lines.append(f"  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
                lines.append(f"  Predicted: {result['prediction']} ({result['confidence']:.2%})")
                lines.append(f"  Ground Truth: Class {ground_truth[idx]}")
                lines.append(f"  Probabilities: {result['all_probabilities']}")
    
    lines.append("")
    lines.append("-" * 40)
    lines.append("LOW CONFIDENCE PREDICTIONS")
    lines.append("-" * 40)
    
    low_conf = [(i, r) for i, r in enumerate(results) if r['confidence'] < 0.7]
    for idx, result in sorted(low_conf, key=lambda x: x[1]['confidence']):
        lines.append(f"\nSequence {idx}: {result['prediction']} ({result['confidence']:.2%})")
        lines.append(f"  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
        lines.append(f"  Probabilities: {result['all_probabilities']}")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model predictions with attention')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--time', type=float, default=None, help='Start time in seconds (random if not specified)')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--classes', type=str, nargs='+', default=['Non-Violence', 'Violence'],
                       help='Class names')
    parser.add_argument('--cmap', type=str, default='Greys',
                       help='Colormap for attention (Greys, Blues, Reds, etc.)')
    
    args = parser.parse_args()
    
    result = visualize_prediction_with_attention(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        video_path=args.video,
        start_time=args.time,
        save_path=args.save,
        class_names=args.classes,
        attention_cmap=args.cmap
    )
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for key, value in result.items():
        if key not in ['mean_attention_per_frame']:
            print(f"{key}: {value}")