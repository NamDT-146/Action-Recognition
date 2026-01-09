import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
from abc import ABC, abstractmethod


class FrameLevelTransform(ABC):
    """
    Abstract base class for frame-level transforms. 
    Its forward method applies on image level.
    Each subclass represents a specific frame-level transform.
    Optional arguments can be passed during initialization.
    """
    
    @abstractmethod
    def forward(self, frame):
        """
        Apply transform to a single frame.
        
        Args:
            frame: Single frame (numpy array or tensor)
            
        Returns:
            Transformed frame
        """
        pass
    
    def __call__(self, frame):
        return self.forward(frame)


class VideoLevelTransform(ABC):
    """
    Abstract base class for video-level transforms. 
    Its forward method applies on video sequence level.
    Each subclass represents a specific video-level transform.
    Optional arguments can be passed during initialization.
    """
    
    @abstractmethod
    def forward(self, frames):
        """
        Apply transform to a sequence of frames.
        
        Args:
            frames: List of frames or tensor of shape (T, C, H, W)
            
        Returns:
            Transformed sequence
        """
        pass
    
    def __call__(self, frames):
        # print("Shape/type before", self.__class__.__name__, ":",
        #       frames.shape if isinstance(frames, torch.Tensor) else type(frames))
        # print("Applying", self.__class__.__name__)
        return self.forward(frames)


# ==================== Frame-Level Transforms ====================

class ExtractFrames(FrameLevelTransform):
    """Extract frames from video file."""
    
    def __init__(self, num_frames=21):
        """
        Args:
            num_frames: Number of consecutive frames to extract
        """
        self.num_frames = num_frames
    
    def forward(self, video_path):
        """
        Extract consecutive frames from video.
        
        Logic:
        - If video has fewer frames than required: load all frames and duplicate until enough
        - If video has enough frames: choose random starting position and extract num_frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames (numpy arrays) with length = num_frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle invalid video
        if total_frames <= 0:
            cap.release()
            # print(f"Warning: Could not read total frames from {video_path}")
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return [dummy_frame] * self.num_frames
        
        # Extract frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Handle case: fewer frames than required
        if len(frames) < self.num_frames:
            # Duplicate frames until we have enough
            original_frames = frames.copy()
            while len(frames) < self.num_frames:
                frames.extend(original_frames)
            return frames[:self.num_frames]
        
        # Handle case: enough frames available
        # Choose random starting position
        max_start = total_frames - self.num_frames
        start_frame = random.randint(0, max_start)
        return frames[start_frame:start_frame + self.num_frames]


class BGRToRGB(FrameLevelTransform):
    """Convert BGR frame to RGB."""
    
    def forward(self, frame):
        """
        Convert BGR to RGB.
        
        Args:
            frame: BGR frame (numpy array)
            
        Returns:
            RGB frame (numpy array)
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class CropDarkBorders(FrameLevelTransform):
    """Remove dark borders from frame."""
    
    def __init__(self, crop_coords):
        """
        Args:
            crop_coords: Tuple (x_crop, y_crop) for border removal
        """
        self.crop_coords = crop_coords
    
    def forward(self, frame):
        """
        Remove dark borders.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Cropped frame (numpy array)
        """
        if self.crop_coords is None:
            return frame
        
        x_crop, y_crop = self.crop_coords
        h, w = frame.shape[:2]
        return frame[y_crop:h - y_crop, x_crop:w - x_crop]


class RandomCrop(FrameLevelTransform):
    """Apply random crop based on corner position."""
    
    def __init__(self, crop_percentage=0.8, corner=None):
        """
        Args:
            crop_percentage: Percentage of image to crop (0.8 = 80%)
            corner: Corner position ("Center", "Left_up", "Left_down", "Right_up", "Right_down")
                   If None, randomly selects from all corners
                   If list, randomly selects from the provided corners
        """
        self.crop_percentage = crop_percentage
        self.available_corners = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]
        
        if corner is None:
            # Randomly choose from all corners
            self.corner = random.choice(self.available_corners)
        elif isinstance(corner, list):
            # Randomly choose from provided corners
            self.corner = random.choice(corner)
        else:
            # Use specified corner
            self.corner = corner
        
        # Store crop coordinates (will be set on first frame)
        self._crop_coords = None
    
    def _get_crop_coords(self, h, w):
        """Calculate crop coordinates based on corner and frame dimensions."""
        crop_size = int(min(h, w) * self.crop_percentage)
        
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
        """
        Apply crop based on corner.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Cropped frame (numpy array)
        """
        h, w = frame.shape[:2]
        
        # Calculate and cache crop coordinates on first call
        if self._crop_coords is None:
            self._crop_coords = self._get_crop_coords(h, w)
        
        x_start, y_start, crop_size = self._crop_coords
        return frame[y_start:y_start + crop_size, x_start:x_start + crop_size]
    
    def reset(self):
        """Reset cached crop coordinates for new sequence."""
        self._crop_coords = None



class ResizeFrame(FrameLevelTransform):
    """Resize frame to target size."""
    
    def __init__(self, size=224):
        """
        Args:
            size: Target size (width, height)
        """
        self.size = (size, size) if isinstance(size, int) else size
    
    def forward(self, frame):
        """
        Resize frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Resized frame (numpy array)
        """
        return cv2.resize(frame, self.size)


class NormalizeFrame(FrameLevelTransform):
    """Normalize frame with mean and std."""
    
    def __init__(self, mean=None, std=None):
        """
        Args:
            mean: Mean values for normalization (default: ImageNet mean)
            std: Std values for normalization (default: ImageNet std)
        """
        self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406])
        self.std = np.array(std if std is not None else [0.229, 0.224, 0.225])
    
    def forward(self, frame):
        """
        Normalize frame.
        
        Args:
            frame: Input frame (numpy array, 0-255 or 0-1)
            
        Returns:
            Normalized frame (numpy array)
        """
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
        return (frame - self.mean) / self.std


class ToTensor(FrameLevelTransform):
    """Convert numpy array to PyTorch tensor."""
    
    def forward(self, frame):
        """
        Convert to tensor and permute to (C, H, W).
        
        Args:
            frame: Input frame (numpy array, H, W, C)
            
        Returns:
            Tensor (C, H, W)
        """
        if isinstance(frame, np.ndarray):
            return torch.from_numpy(frame).permute(2, 0, 1).float()
        return frame


class ToPILImage(FrameLevelTransform):
    """Convert tensor or numpy array to PIL Image."""
    
    def forward(self, frame):
        """
        Convert to PIL Image.
        
        Args:
            frame: Input frame (numpy array or tensor)
            
        Returns:
            PIL Image
        """
        return transforms.ToPILImage()(frame)


# ==================== Video-Level Transforms ====================

class StackFrames(VideoLevelTransform):
    """Stack list of frames into a single tensor."""
    
    def forward(self, frames):
        """
        Stack frames.
        
        Args:
            frames: List of frame tensors
            
        Returns:
            Stacked tensor (T, C, H, W)
        """
        return torch.stack(frames).float()

class ComputeFrameDifferences(VideoLevelTransform):
    """Compute temporal differences between consecutive frames."""
    
    def forward(self, frames):
        """
        Compute differences between consecutive frames.
        
        Args:
            frames: List of frame tensors or stacked tensor (T, C, H, W)
            
        Returns:
            Tensor of frame differences (T-1, C, H, W)
        """
        if isinstance(frames, list):
            frames = torch.stack(frames)
        
        frame_diffs = []
        for i in range(len(frames) - 1):
            diff = frames[i] - frames[i + 1]
            frame_diffs.append(diff)
        
        return torch.stack(frame_diffs).float()


class TemporalSubsampling(VideoLevelTransform):
    """Subsample frames temporally."""
    
    def __init__(self, num_frames=20):
        """
        Args:
            num_frames: Target number of frames
        """
        self.num_frames = num_frames
    
    def forward(self, frames):
        """
        Subsample frames.
        
        Args:
            frames: List of frames or tensor
            
        Returns:
            Subsampled frames
        """
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
        """
        Transpose sequence dimensions.
        
        Args:
            frames: Tensor of shape (S, C, H, W) where S is sequence length
            
        Returns:
            Tensor of shape (C, S, H, W)
        """
        # print("Before transpose, frames shape:", frames.shape)
        if isinstance(frames, torch.Tensor):
            # Input: (S, C, H, W) -> Output: (C, S, H, W)
            # print("After transpose, frames shape:", frames.shape)
            return frames.permute(1, 0, 2, 3)
        return frames


# ==================== Main VideoTransforms Class ====================

class VideoTransforms:
    """
    Main class for video transforms. Combines frame-level and video-level transforms.
    
    The forward method applies the sequence of transforms to a list of frames (video sequence).
    Video transforms are applied after frame-level transforms.
    """
    
    def __init__(self, frame_transforms=None, video_transforms=None):
        """
        Args:
            frame_transforms: List of FrameLevelTransform instances
            video_transforms: List of VideoLevelTransform instances
        """
        self.frame_transforms = frame_transforms if frame_transforms is not None else []
        self.video_transforms = video_transforms if video_transforms is not None else []
    
    def forward(self, frames):
        """
        Apply all transforms to the video sequence.
        
        Args:
            frames: List of frames or video path (if ExtractFrames is first transform)
            
        Returns:
            Transformed video sequence (tensor)
        """
        # Apply frame-level transforms
        processed_frames = frames
        for transform in self.frame_transforms:
            if isinstance(transform, ExtractFrames):
                # Extract frames from video path
                processed_frames = transform(processed_frames)
            else:
                # Apply to each frame
                if isinstance(processed_frames, list):
                    processed_frames = [transform(frame) for frame in processed_frames]
                else:
                    processed_frames = transform(processed_frames)
        
        # Apply video-level transforms
        for transform in self.video_transforms:
            # print("Before", transform.__class__.__name__, "shape/type:",
            #       processed_frames.shape if isinstance(processed_frames, torch.Tensor) else type(processed_frames))
            processed_frames = transform(processed_frames)
            # print(transform.__class__.__name__, "applied. Current shape/type:",
            #       processed_frames.shape if isinstance(processed_frames, torch.Tensor) else type(processed_frames))
        
        return processed_frames
    
    def __call__(self, frames):
        return self.forward(frames)
    
    @staticmethod
    def get_default_transform(figure_size=224, seq_length=20, crop_dark=None, crop_percentage=0.8, model_name=None):
        """
        Returns a default video transform pipeline.
        
        Args:
            figure_size: Target frame size
            seq_length: Number of frame differences
            crop_dark: Coordinates for dark border removal
            crop_percentage: Crop percentage for random crop
            
        Returns:
            VideoTransforms instance with default pipeline
        """
        frame_transforms = [
            ExtractFrames(num_frames=seq_length + 1),  # Need seq_length+1 frames for seq_length differences
            BGRToRGB(),
        ]
        
        if crop_dark is not None:
            frame_transforms.append(CropDarkBorders(crop_coords=crop_dark))
        
        # Note: RandomCrop corner will be set per video in dataset
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
                                   crop_percentage=0.8, crop_corner=None, model_name=None):
        """
        Returns a preprocessing transform for a specific video with fixed crop corner.
        
        Args:
            figure_size: Target frame size
            seq_length: Number of frame differences
            crop_dark: Coordinates for dark border removal
            crop_percentage: Crop percentage
            crop_corner: Specific corner for cropping
            
        Returns:
            VideoTransforms instance with preprocessing pipeline
        """
        frame_transforms = [
            ExtractFrames(num_frames=seq_length),
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
        """
        Extend the frame-level transforms with additional transforms.
        
        Args:
            new_transforms: List of FrameLevelTransform instances to add
        """
        self.frame_transforms.extend(new_transforms)

    def extend_video_transforms(self, new_transforms):
        """
        Extend the video-level transforms with additional transforms.
        
        Args:
            new_transforms: List of VideoLevelTransform instances to add
        """
        self.video_transforms.extend(new_transforms)

# ==================== Utility Functions ====================

def get_torchvision_transform(figure_size=224):
    """
    Returns a standard torchvision transform for comparison.
    
    Args:
        figure_size: Target size for resizing
        
    Returns:
        torchvision.transforms.Compose instance
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((figure_size, figure_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])