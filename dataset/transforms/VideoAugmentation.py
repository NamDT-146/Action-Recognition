"""
Video Augmentation transforms for MIL training.
Includes temporal, spatial, and photometric augmentations.
"""

import torch
import cv2
import numpy as np
import random
from typing import List, Tuple


class VideoAugmentation:
    """Base class for video-level augmentations."""
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentation.
        
        Args:
            frames: numpy array of shape (T, H, W, 3) in BGR or RGB
            
        Returns:
            Augmented frames of same shape
        """
        raise NotImplementedError


# ==================== Spatial Augmentations ====================

class RandomHorizontalFlip(VideoAugmentation):
    """Randomly flip video horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            frames = np.ascontiguousarray(frames[:, :, ::-1, :])  # Flip along width
        return frames


class RandomVerticalFlip(VideoAugmentation):
    """Randomly flip video vertically."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            frames = np.ascontiguousarray(frames[:, ::-1, :, :])  # Flip along height
        return frames


class RandomRotation(VideoAugmentation):
    """Randomly rotate video."""
    
    def __init__(self, degrees: Tuple[float, float] = (-10, 10), p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            h, w = frames.shape[1:3]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to each frame
            rotated_frames = []
            for frame in frames:
                rotated = cv2.warpAffine(frame, mat, (w, h),
                                        borderMode=cv2.BORDER_REFLECT_101)
                rotated_frames.append(rotated)
            
            frames = np.array(rotated_frames)
        
        return frames


class RandomAffine(VideoAugmentation):
    """Apply random affine transformation."""
    
    def __init__(self, scale: Tuple[float, float] = (0.8, 1.2),
                 translate: Tuple[float, float] = (-0.1, 0.1),
                 p: float = 0.5):
        self.scale = scale
        self.translate = translate
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            h, w = frames.shape[1:3]
            
            # Random scale
            scale = random.uniform(self.scale[0], self.scale[1])
            
            # Random translation
            tx = random.uniform(self.translate[0], self.translate[1]) * w
            ty = random.uniform(self.translate[0], self.translate[1]) * h
            
            # Build affine matrix
            mat = np.array([
                [scale, 0, tx],
                [0, scale, ty]
            ], dtype=np.float32)
            
            # Apply to each frame
            transformed_frames = []
            for frame in frames:
                transformed = cv2.warpAffine(frame, mat, (w, h),
                                            borderMode=cv2.BORDER_REFLECT_101)
                transformed_frames.append(transformed)
            
            frames = np.array(transformed_frames)
        
        return frames


class RandomCropResize(VideoAugmentation):
    """Randomly crop and resize (RandAugment style)."""
    
    def __init__(self, scale: Tuple[float, float] = (0.8, 1.0),
                 ratio: Tuple[float, float] = (3./4., 4./3.),
                 p: float = 0.5):
        self.scale = scale
        self.ratio = ratio
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            h, w = frames.shape[1:3]
            
            # Random scale and aspect ratio
            scale = random.uniform(self.scale[0], self.scale[1])
            ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            crop_h = int(h * scale)
            crop_w = int(crop_h * ratio)
            
            # Ensure within bounds
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)
            
            # Random position
            top = random.randint(0, h - crop_h) if h > crop_h else 0
            left = random.randint(0, w - crop_w) if w > crop_w else 0
            
            # Crop and resize
            cropped_frames = []
            for frame in frames:
                cropped = frame[top:top+crop_h, left:left+crop_w]
                resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                cropped_frames.append(resized)
            
            frames = np.array(cropped_frames)
        
        return frames


class RandomPerspective(VideoAugmentation):
    """Apply random perspective transformation."""
    
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            h, w = frames.shape[1:3]
            
            # Define random perspective points
            half_h = h / 2
            half_w = w / 2
            distortion = self.distortion_scale
            
            src_points = np.array([
                [0, 0],
                [w, 0],
                [0, h],
                [w, h]
            ], dtype=np.float32)
            
            dst_points = src_points + np.random.uniform(
                -distortion * w, distortion * w, src_points.shape
            ).astype(np.float32)
            
            # Get perspective matrix
            mat = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply to each frame
            transformed_frames = []
            for frame in frames:
                transformed = cv2.warpPerspective(frame, mat, (w, h),
                                                 borderMode=cv2.BORDER_REFLECT_101)
                transformed_frames.append(transformed)
            
            frames = np.array(transformed_frames)
        
        return frames


class GaussianBlur(VideoAugmentation):
    """Apply Gaussian blur."""
    
    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0),
                 p: float = 0.5):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            blurred_frames = []
            
            for frame in frames:
                blurred = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), sigma)
                blurred_frames.append(blurred)
            
            frames = np.array(blurred_frames)
        
        return frames


# ==================== Photometric Augmentations ====================

class ColorJitter(VideoAugmentation):
    """Randomly adjust brightness, contrast, saturation, hue."""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            # Generate random factors (same for entire video for consistency)
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            hue_shift = random.uniform(-self.hue, self.hue)
            
            jittered_frames = []
            for frame in frames:
                # Assume BGR format
                adjusted = frame.astype(np.float32)
                
                # Brightness
                adjusted = adjusted * brightness_factor
                
                # Contrast
                adjusted = adjusted * contrast_factor + 127 * (1 - contrast_factor)
                
                # Saturation and Hue (convert to HSV)
                hsv = cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor  # Saturation
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180  # Hue
                adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                # Clip values
                adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
                jittered_frames.append(adjusted)
            
            frames = np.array(jittered_frames)
        
        return frames


class RandomBrightness(VideoAugmentation):
    """Randomly adjust brightness."""
    
    def __init__(self, delta: float = 0.3, p: float = 0.5):
        self.delta = delta
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            factor = random.uniform(1 - self.delta, 1 + self.delta)
            frames = (frames.astype(np.float32) * factor).astype(np.uint8)
        return frames


class RandomContrast(VideoAugmentation):
    """Randomly adjust contrast."""
    
    def __init__(self, delta: float = 0.3, p: float = 0.5):
        self.delta = delta
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            factor = random.uniform(1 - self.delta, 1 + self.delta)
            frames = (frames.astype(np.float32) - 127) * factor + 127
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        return frames


class RandomSaturation(VideoAugmentation):
    """Randomly adjust saturation."""
    
    def __init__(self, delta: float = 0.5, p: float = 0.5):
        self.delta = delta
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            factor = random.uniform(1 - self.delta, 1 + self.delta)
            saturated_frames = []
            
            for frame in frames:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
                saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                saturated_frames.append(saturated)
            
            frames = np.array(saturated_frames)
        
        return frames


class RandomGrayscale(VideoAugmentation):
    """Randomly convert to grayscale."""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            gray_frames = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_frames.append(gray_frame)
            frames = np.array(gray_frames)
        
        return frames


class RandomNoise(VideoAugmentation):
    """Add random Gaussian noise."""
    
    def __init__(self, noise_std: float = 10.0, p: float = 0.3):
        self.noise_std = noise_std
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, frames.shape).astype(np.float32)
            frames = np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return frames


class RandomEqualization(VideoAugmentation):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8, p: float = 0.3):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=(self.tile_size, self.tile_size))
            
            equalized_frames = []
            for frame in frames:
                # Convert BGR to LAB color space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
                # Apply CLAHE to L channel
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                
                # Convert back to BGR
                equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                equalized_frames.append(equalized)
            
            frames = np.array(equalized_frames)
        
        return frames


# ==================== Temporal Augmentations ====================

class TemporalReverse(VideoAugmentation):
    """Randomly reverse video frames."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            frames = frames[::-1].copy()
        return frames


class TemporalDropout(VideoAugmentation):
    """Randomly duplicate frames (temporal dropout simulation)."""
    
    def __init__(self, max_drop_frames: int = 2, p: float = 0.3):
        self.max_drop_frames = max_drop_frames
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p and len(frames) > 1:
            num_frames = len(frames)
            drop_frames = random.randint(1, self.max_drop_frames)
            drop_indices = random.sample(range(num_frames), min(drop_frames, num_frames - 1))
            
            # Replace dropped frames with duplicates of adjacent frame
            for idx in drop_indices:
                if idx > 0:
                    frames[idx] = frames[idx - 1].copy()
                else:
                    frames[idx] = frames[idx + 1].copy()
        
        return frames


class TemporalShift(VideoAugmentation):
    """Temporal shift of frames (for motion augmentation)."""
    
    def __init__(self, max_shift: int = 2, p: float = 0.3):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                frames = np.roll(frames, shift, axis=0)
        
        return frames


# ==================== Augmentation Pipeline ====================

class VideoAugmentationPipeline:
    """Compose multiple augmentations."""
    
    def __init__(self, augmentations: List[VideoAugmentation] = None):
        self.augmentations = augmentations or []
    
    def add(self, augmentation: VideoAugmentation) -> 'VideoAugmentationPipeline':
        self.augmentations.append(augmentation)
        return self
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        for augmentation in self.augmentations:
            frames = augmentation(frames)
        return frames
    
    @staticmethod
    def create_light_augmentation() -> 'VideoAugmentationPipeline':
        """Light augmentation for sensitive tasks."""
        return VideoAugmentationPipeline([
            RandomHorizontalFlip(p=0.5),
            RandomBrightness(delta=0.1, p=0.3),
            RandomContrast(delta=0.1, p=0.3),
        ])
    
    @staticmethod
    def create_medium_augmentation() -> 'VideoAugmentationPipeline':
        """Medium augmentation (recommended)."""
        return VideoAugmentationPipeline([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            RandomRotation(degrees=(-5, 5), p=0.3),
            RandomCropResize(scale=(0.8, 1.0), p=0.3),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            RandomBrightness(delta=0.15, p=0.2),
            GaussianBlur(p=0.2),
            RandomNoise(noise_std=5.0, p=0.1),
            TemporalReverse(p=0.2),
        ])
    
    @staticmethod
    def create_strong_augmentation() -> 'VideoAugmentationPipeline':
        """Strong augmentation for large datasets."""
        return VideoAugmentationPipeline([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=(-15, 15), p=0.4),
            RandomAffine(scale=(0.7, 1.3), translate=(-0.15, 0.15), p=0.4),
            RandomCropResize(scale=(0.6, 1.0), p=0.5),
            RandomPerspective(distortion_scale=0.3, p=0.3),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            RandomBrightness(delta=0.2, p=0.3),
            RandomSaturation(delta=0.3, p=0.3),
            GaussianBlur(p=0.3),
            RandomEqualization(p=0.2),
            RandomNoise(noise_std=10.0, p=0.2),
            RandomGrayscale(p=0.1),
            TemporalReverse(p=0.3),
            TemporalDropout(max_drop_frames=2, p=0.2),
            TemporalShift(max_shift=2, p=0.2),
        ])

class FeatureAugmentation:
    """Augmentation strategies for precomputed feature vectors."""
    
    @staticmethod
    def add_gaussian_noise(features: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
        """
        Add Gaussian noise to feature vectors (feature jitter).
        
        Args:
            features: Feature tensor of shape (..., feature_dim)
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Augmented features with added noise
        """
        noise = torch.randn_like(features) * noise_std
        return features + noise
    
    @staticmethod
    def feature_dropout(features: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly zero out feature dimensions (feature dropout).
        
        Args:
            features: Feature tensor of shape (..., feature_dim)
            dropout_rate: Fraction of features to drop (0.0 to 1.0)
            
        Returns:
            Features with random dimensions zeroed
        """
        mask = torch.rand_like(features) > dropout_rate
        return features * mask
    
    @staticmethod
    def mixup_features(features1: torch.Tensor, features2: torch.Tensor,
                      alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix two feature vectors (mixup augmentation).
        
        Args:
            features1: First feature tensor
            features2: Second feature tensor
            alpha: Mixing coefficient (0.0 to 1.0)
            
        Returns:
            Tuple of mixed features
        """
        lam = np.random.beta(alpha, alpha)
        mixed1 = lam * features1 + (1 - lam) * features2
        mixed2 = (1 - lam) * features1 + lam * features2
        return mixed1, mixed2
    
    @staticmethod
    def temporal_shuffle(features: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
        """
        Randomly shuffle temporal order of segments (temporal augmentation).
        
        Args:
            features: Feature tensor of shape (num_segments, feature_dim)
            max_shift: Maximum frame shift
            
        Returns:
            Temporally shuffled features
        """
        if features.shape[0] <= 1:
            return features
        
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            features = torch.roll(features, shift, dims=0)
        
        return features
    
    @staticmethod
    def temporal_dropout(features: torch.Tensor, dropout_rate: float = 0.2) -> torch.Tensor:
        """
        Drop random temporal segments (simulate missing frames).
        
        Args:
            features: Feature tensor of shape (num_segments, feature_dim)
            dropout_rate: Fraction of segments to drop
            
        Returns:
            Features with some segments zeroed
        """
        num_segments = features.shape[0]
        num_drop = max(1, int(num_segments * dropout_rate))
        
        # Randomly select segments to drop
        drop_indices = random.sample(range(num_segments), num_drop)
        
        # Replace with previous segment or zero
        for idx in drop_indices:
            if idx > 0:
                features[idx] = features[idx - 1].clone()
            else:
                features[idx] = torch.zeros_like(features[idx])
        
        return features

# Test
if __name__ == "__main__":
    print("Video Augmentation Module Ready!")
    
    # Create augmentation pipelines
    light_aug = VideoAugmentationPipeline.create_light_augmentation()
    medium_aug = VideoAugmentationPipeline.create_medium_augmentation()
    strong_aug = VideoAugmentationPipeline.create_strong_augmentation()
    
    print(f"Light augmentations: {len(light_aug.augmentations)}")
    print(f"Medium augmentations: {len(medium_aug.augmentations)}")
    print(f"Strong augmentations: {len(strong_aug.augmentations)}")
    
    # Create dummy video
    dummy_video = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    
    print("\nTesting augmentations on dummy video (16, 224, 224, 3)...")
    
    # Test light
    aug_light = light_aug(dummy_video)
    print(f"After light augmentation: {aug_light.shape}")
    
    # Test medium
    aug_medium = medium_aug(dummy_video)
    print(f"After medium augmentation: {aug_medium.shape}")
    
    # Test strong
    aug_strong = strong_aug(dummy_video)
    print(f"After strong augmentation: {aug_strong.shape}")
    
    print("\nAll augmentations working!")

