import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
from abc import ABC, abstractmethod


class FrameLevelTransform(ABC):
    """
    Abstract base class for frame-level transforms. 
    """
    
    @abstractmethod
    def forward(self, frame):
        pass
    
    def __call__(self, frame):
        return self.forward(frame)


class VideoLevelTransform(ABC):
    """
    Abstract base class for video-level transforms. 
    """
    
    @abstractmethod
    def forward(self, frames):
        pass
    
    def __call__(self, frames):
        return self.forward(frames)


# ==================== Frame-Level Transforms ====================

class ExtractFrames(FrameLevelTransform):
    """Extract frames from video file with optional temporal subsampling."""
    
    def __init__(self, num_frames=21, frame_step=1):
        self.num_frames = num_frames
        self.frame_step = frame_step
    
    def forward(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # We don't rely solely on CAP_PROP_FRAME_COUNT as it can be inaccurate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate required span with frame_step
        required_span = (self.num_frames - 1) * self.frame_step + 1
        
        # Extract all frames first
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Explicitly check for valid frame content
            if frame is not None and frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                all_frames.append(frame)
        
        cap.release()
        
        # Handle case: no valid frames found
        if not all_frames:
            print(f"⚠️  Warning: No valid frames found in {video_path}")
            # Return dummy black frames to prevent pipeline crash
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return [dummy_frame] * self.num_frames
        
        # ✅ MODIFIED: If not enough frames, duplicate until we have enough
        if len(all_frames) < required_span:
            print(f"⚠️  Warning: Video has {len(all_frames)} frames but requires {required_span}. Duplicating frames...")
            
            # Duplicate frames until we have enough
            duplicated_frames = all_frames.copy()
            while len(duplicated_frames) < required_span:
                duplicated_frames.extend(all_frames)
            
            all_frames = duplicated_frames[:required_span + (required_span - 1)]  # Add some extra for random selection
            print(f"✅ Duplicated to {len(all_frames)} frames")
        
        # Now all_frames has at least required_span frames
        # Choose a random segment
        max_start = len(all_frames) - required_span
        start_frame = random.randint(0, max_start)
        
        # Extract frames with step
        frame_indices = [start_frame + i * self.frame_step for i in range(self.num_frames)]
        return [all_frames[i] for i in frame_indices]


class BGRToRGB(FrameLevelTransform):
    """Convert BGR frame to RGB."""
    def forward(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class CropDarkBorders(FrameLevelTransform):
    """Remove dark borders from frame."""
    def __init__(self, crop_coords):
        self.crop_coords = crop_coords
    
    def forward(self, frame):
        if self.crop_coords is None:
            return frame
        
        x_crop, y_crop = self.crop_coords
        h, w = frame.shape[:2]
        
        # Safety check: if crop is too aggressive
        if y_crop * 2 >= h or x_crop * 2 >= w:
            return frame
            
        return frame[y_crop:h - y_crop, x_crop:w - x_crop]


class RandomCrop(FrameLevelTransform):
    """Apply random crop based on corner position."""
    
    def __init__(self, crop_percentage=0.8, corner=None):
        self.crop_percentage = crop_percentage
        self.available_corners = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        if corner is None:
            self.corner = random.choice(self.available_corners)
        elif isinstance(corner, list):
            self.corner = random.choice(corner)
        else:
            self.corner = corner
        
        self._crop_coords = None
    
    def _get_crop_coords(self, h, w):
        crop_size = int(min(h, w) * self.crop_percentage)
        # Ensure at least 1 pixel
        crop_size = max(1, crop_size)
        
        if self.corner == "Left_up":
            x_start, y_start = 0, 0
        elif self.corner == "Right_down":
            x_start, y_start = w - crop_size, h - crop_size
        elif self.corner == "Right_up":
            x_start, y_start = w - crop_size, 0
        elif self.corner == "Left_down":
            x_start, y_start = 0, h - crop_size
        else:  # Center
            x_start, y_start = (w - crop_size) // 2, (h - crop_size) // 2
        
        x_start = max(0, min(x_start, w - crop_size))
        y_start = max(0, min(y_start, h - crop_size))
        
        return x_start, y_start, crop_size
    
    def forward(self, frame):
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame
        
        if self._crop_coords is None:
            self._crop_coords = self._get_crop_coords(h, w)
        
        x_start, y_start, crop_size = self._crop_coords
        
        cropped = frame[y_start:y_start + crop_size, x_start:x_start + crop_size]
        
        # Final safety check
        if cropped.size == 0:
            return frame
            
        return cropped
    
    def reset(self):
        self._crop_coords = None


class ResizeFrame(FrameLevelTransform):
    """Resize frame to target size with safety checks."""
    
    def __init__(self, size=224):
        self.size = (size, size) if isinstance(size, int) else size
    
    def forward(self, frame):
        # ⚠️ CRITICAL FIX: Check for empty frames/invalid dimensions provided by previous steps
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            # print(f"⚠️  ResizeFrame received bad frame shape: {frame.shape if frame is not None else 'None'}. Returning black frame.")
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        try:
            return cv2.resize(frame, self.size)
        except Exception as e:
            print(f"❌ ResizeFrame Failed. Input shape: {frame.shape}, Error: {e}")
            # Fallback to prevent dataset crash
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)


class NormalizeFrame(FrameLevelTransform):
    """Normalize frame with mean and std."""
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406])
        self.std = np.array(std if std is not None else [0.229, 0.224, 0.225])
    
    def forward(self, frame):
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
        return (frame - self.mean) / self.std


class ToTensor(FrameLevelTransform):
    """Convert numpy array to PyTorch tensor."""
    def forward(self, frame):
        if isinstance(frame, np.ndarray):
            return torch.from_numpy(frame).permute(2, 0, 1).float()
        return frame


class ToPILImage(FrameLevelTransform):
    def forward(self, frame):
        return transforms.ToPILImage()(frame)


# ==================== Video-Level Transforms ====================

class StackFrames(VideoLevelTransform):
    """Stack list of frames into a single tensor."""
    def forward(self, frames):
        return torch.stack(frames).float()


class ComputeFrameDifferences(VideoLevelTransform):
    """Compute temporal differences between consecutive frames."""
    def forward(self, frames):
        if isinstance(frames, list):
            frames = torch.stack(frames)
        
        frame_diffs = []
        for i in range(len(frames) - 1):
            diff = frames[i] - frames[i + 1]
            frame_diffs.append(diff)
        
        return torch.stack(frame_diffs).float()


class TemporalSubsampling(VideoLevelTransform):
    def __init__(self, num_frames=20):
        self.num_frames = num_frames
    
    def forward(self, frames):
        if isinstance(frames, torch.Tensor):
            total_frames = frames.shape[0]
        else:
            total_frames = len(frames)
        
        if total_frames <= self.num_frames:
            return frames
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        if isinstance(frames, torch.Tensor):
            return frames[indices]
        else:
            return [frames[i] for i in indices]


class TransposeSequence(VideoLevelTransform):
    """Transpose video sequence from [B,S,C,H,W] to [B,C,S,H,W]."""
    def forward(self, frames):
        if isinstance(frames, torch.Tensor):
            return frames.permute(1, 0, 2, 3)
        return frames


# ==================== Main VideoTransforms Class ====================

class VideoTransforms:
    def __init__(self, frame_transforms=None, video_transforms=None):
        self.frame_transforms = frame_transforms if frame_transforms is not None else []
        self.video_transforms = video_transforms if video_transforms is not None else []
    
    def forward(self, frames):
        # Apply frame-level transforms
        processed_frames = frames
        for transform in self.frame_transforms:
            if isinstance(transform, ExtractFrames):
                processed_frames = transform(processed_frames)
            else:
                if isinstance(processed_frames, list):
                    # For transforms like RandomCrop, ensure consistency across sequence if needed
                    if hasattr(transform, 'reset'):
                        transform.reset()
                    processed_frames = [transform(frame) for frame in processed_frames]
                else:
                    processed_frames = transform(processed_frames)
        
        # Apply video-level transforms
        for transform in self.video_transforms:
            processed_frames = transform(processed_frames)
        
        return processed_frames
    
    def __call__(self, frames):
        return self.forward(frames)
    
    @staticmethod
    def get_default_transform(figure_size=224, seq_length=20, crop_dark=None, 
                             crop_percentage=0.8, frame_step=1, model_name=None):
        frame_transforms = [
            ExtractFrames(num_frames=seq_length + 1, frame_step=frame_step),
            BGRToRGB(),
        ]
        
        if crop_dark is not None:
            frame_transforms.append(CropDarkBorders(crop_coords=crop_dark))
        
        frame_transforms.extend([
            ResizeFrame(size=figure_size),
            NormalizeFrame(),
            ToTensor(),
        ])
        
        video_transforms = [
            ComputeFrameDifferences()
        ]

        if model_name == "I3D":
            video_transforms.append(TransposeSequence())

        return VideoTransforms(
            frame_transforms=frame_transforms,
            video_transforms=video_transforms
        )
    
    @staticmethod
    def get_preprocessing_transform(figure_size=224, seq_length=20, crop_dark=None, 
                                   crop_percentage=0.8, crop_corner=None, frame_step=1, model_name=None):
        frame_transforms = [
            ExtractFrames(num_frames=seq_length, frame_step=frame_step),
            BGRToRGB(),
        ]
        
        if crop_dark is not None:
            frame_transforms.append(CropDarkBorders(crop_coords=crop_dark))
        
        frame_transforms.extend([
            RandomCrop(crop_percentage=crop_percentage, corner=crop_corner),
            ResizeFrame(size=figure_size),
            NormalizeFrame(),
            ToTensor(),
        ])
        
        video_transforms = [
            StackFrames()
        ]
        
        if model_name == "I3D":
            video_transforms.append(TransposeSequence())

        return VideoTransforms(
            frame_transforms=frame_transforms,
            video_transforms=video_transforms
        )

    def extend_frame_transforms(self, new_transforms):
        self.frame_transforms.extend(new_transforms)

    def extend_video_transforms(self, new_transforms):
        self.video_transforms.extend(new_transforms)


def get_torchvision_transform(figure_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((figure_size, figure_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])