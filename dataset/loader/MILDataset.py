"""
Multiple Instance Learning (MIL) Dataset for Weakly-Supervised Anomaly Detection.
Implements paired dataset loading (Normal/Abnormal) with segment-based video processing.
"""

import os
import torch
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from dataset.transforms.VideoAugmentation import VideoAugmentationPipeline, FeatureAugmentation

class PairedMILVideoDataset(Dataset):
    """
    Paired Dataset for MIL training with augmentation support.
    Returns (Normal_Bag, Abnormal_Bag) where each bag contains multiple segments.
    """
    
    def __init__(
        self,
        normal_dir: str,
        abnormal_dir: str,
        num_frames: int = 16,
        frame_step: int = 2,
        num_segments: int = 5,
        img_size: int = 224,
        min_segment_step_factor: float = 0.33,
        video_extensions: Tuple[str] = ('.avi', '.mp4', '.mpg', '.mov'),
        augmentation: str = 'medium',
        augmentation_pipeline: Optional = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            normal_dir: Directory containing normal videos
            abnormal_dir: Directory containing abnormal videos
            num_frames: Number of frames per segment (T)
            frame_step: Step between frames within segment
            num_segments: Number of segments per video bag
            img_size: Target frame size
            min_segment_step_factor: Minimum segment step factor
            video_extensions: Valid video extensions
            augmentation: 'light', 'medium', 'strong', or None
            augmentation_pipeline: Custom augmentation pipeline (overrides augmentation param)
            seed: Random seed
        """
        self.normal_dir = Path(normal_dir)
        self.abnormal_dir = Path(abnormal_dir)
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.num_segments = num_segments
        self.img_size = img_size
        self.min_segment_step_factor = min_segment_step_factor
        self.video_extensions = video_extensions
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Setup augmentation
        if augmentation_pipeline is not None:
            self.augmentation = augmentation_pipeline
        elif augmentation == 'light':
            self.augmentation = VideoAugmentationPipeline.create_light_augmentation()
        elif augmentation == 'medium':
            self.augmentation = VideoAugmentationPipeline.create_medium_augmentation()
        elif augmentation == 'strong':
            self.augmentation = VideoAugmentationPipeline.create_strong_augmentation()
        else:
            self.augmentation = None
        
        # Calculate segment span
        self.segment_span = (num_frames - 1) * frame_step + 1
        
        # Load video paths
        self.normal_videos = self._load_video_paths(self.normal_dir)
        self.abnormal_videos = self._load_video_paths(self.abnormal_dir)
        
        if len(self.normal_videos) == 0:
            raise ValueError(f"No normal videos found in {normal_dir}")
        if len(self.abnormal_videos) == 0:
            raise ValueError(f"No abnormal videos found in {abnormal_dir}")
        
        print(f"Loaded {len(self.normal_videos)} normal videos")
        print(f"Loaded {len(self.abnormal_videos)} abnormal videos")
        print(f"Segment config: {num_segments} segments x {num_frames} frames "
              f"(step={frame_step}, span={self.segment_span})")
        if self.augmentation:
            print(f"Augmentation: {len(self.augmentation.augmentations)} transforms")
    
    def _load_video_paths(self, directory: Path) -> List[str]:
        """Load all video paths from directory."""
        videos = []
        for ext in self.video_extensions:
            videos.extend([str(p) for p in directory.glob(f"*{ext}")])
        return sorted(videos)
    
    def _compute_segment_positions(self, total_frames: int) -> List[int]:
        """
        Compute starting positions for segments.
        
        Args:
            total_frames: Total frames in video
            
        Returns:
            List of starting frame indices for each segment
        """
        # Required frames for all segments if placed end-to-end
        min_required = self.num_segments * self.segment_span
        
        if total_frames < self.segment_span:
            raise ValueError(
                f"Video too short: {total_frames} frames < {self.segment_span} required"
            )
        
        # Compute step between segment starts
        max_start = total_frames - self.segment_span
        
        if total_frames < min_required:
            # Video shorter than ideal - compute minimum step
            segment_step = max_start // (self.num_segments - 1) if self.num_segments > 1 else 0
            
            # Check if step is too small
            min_step = int(self.segment_span * self.min_segment_step_factor)
            if segment_step < min_step:
                # Need to duplicate video
                return None  # Signal that duplication is needed
        else:
            # Video long enough - uniform distribution
            segment_step = max_start // (self.num_segments - 1) if self.num_segments > 1 else 0
        
        # Generate positions
        positions = [i * segment_step for i in range(self.num_segments)]
        
        # Ensure last position is valid
        positions[-1] = min(positions[-1], max_start)
        
        return positions
    
    def _extract_segment_frames(
        self,
        video_path: str,
        start_frame: int
    ) -> np.ndarray:
        """
        Extract frames for a single segment.
        
        Args:
            video_path: Path to video
            start_frame: Starting frame index
            
        Returns:
            frames: numpy array of shape (num_frames, H, W, 3) in RGB
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            for i in range(self.num_frames):
                frame_idx = start_frame + i * self.frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    raise IOError(f"Failed to read frame {frame_idx}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame_resized = cv2.resize(
                    frame_rgb, 
                    (self.img_size, self.img_size),
                    interpolation=cv2.INTER_LINEAR
                )
                
                frames.append(frame_resized)
        finally:
            cap.release()
        
        return np.array(frames)
    
    def _load_video_bag(self, video_path: str, apply_augmentation: bool = True) -> torch.Tensor:
        """
        Load video as bag of segments.
        
        Args:
            video_path: Path to video file
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            bag: Tensor of shape (num_segments, 3, num_frames, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Compute segment positions
        positions = self._compute_segment_positions(total_frames)
        
        # Handle video duplication if needed
        if positions is None:
            cap = cv2.VideoCapture(video_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            
            copies_needed = int(np.ceil(self.num_segments * self.segment_span / len(all_frames)))
            extended_frames = all_frames * copies_needed
            
            positions = self._compute_segment_positions(len(all_frames))
            if positions is None:
                positions = [
                    min(i * (total_frames // self.num_segments), total_frames - self.segment_span)
                    for i in range(self.num_segments)
                ]
        
        # Extract segments
        segments = []
        for start_pos in positions:
            try:
                frames = self._extract_segment_frames(video_path, start_pos)
                
                # Convert to tensor (T, H, W, C) -> (C, T, H, W)
                segment_tensor = torch.from_numpy(frames).float()
                segment_tensor = segment_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
                
                # Normalize
                segment_tensor = segment_tensor / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
                segment_tensor = (segment_tensor - mean) / std
                
                segments.append(segment_tensor)
            except Exception as e:
                print(f"Error extracting segment at {start_pos} from {video_path}: {e}")
                segments.append(torch.zeros(3, self.num_frames, self.img_size, self.img_size))
        
        # Stack to (num_segments, C, T, H, W)
        bag = torch.stack(segments)
        
        # Apply augmentation (on video level before normalization if desired)
        if apply_augmentation and self.augmentation is not None:
            # Convert back to numpy for augmentation
            bag_np = (bag.permute(0, 2, 3, 4, 1).numpy() * 255).astype(np.uint8)  # (S, T, H, W, C)
            
            # Apply augmentation to each segment
            augmented_segments = []
            for seg_idx in range(bag_np.shape[0]):
                segment_frames = bag_np[seg_idx]  # (T, H, W, C)
                augmented = self.augmentation(segment_frames)  # Apply augmentation
                augmented_segments.append(augmented)
            
            # Convert back to tensor
            bag_np = np.stack(augmented_segments)
            bag = torch.from_numpy(bag_np.astype(np.float32)).permute(0, 4, 1, 2, 3) / 255.0
            
            # Re-normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            bag = (bag - mean) / std
        
        return bag
    
    def __len__(self):
        """Length determined by abnormal videos (focus on anomalies)."""
        return len(self.abnormal_videos)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get paired normal and abnormal bags with augmentation.
        
        Returns:
            (normal_bag, abnormal_bag):
                Each tensor of shape (num_segments, 3, num_frames, H, W)
        """
        abnormal_path = self.abnormal_videos[idx]
        
        normal_idx = random.randint(0, len(self.normal_videos) - 1)
        normal_path = self.normal_videos[normal_idx]
        
        # Load bags with augmentation
        try:
            bag_normal = self._load_video_bag(normal_path, apply_augmentation=True)
        except Exception as e:
            print(f"Error loading normal video {normal_path}: {e}")
            bag_normal = torch.zeros(
                self.num_segments, 3, self.num_frames, self.img_size, self.img_size
            )
        
        try:
            bag_abnormal = self._load_video_bag(abnormal_path, apply_augmentation=True)
        except Exception as e:
            print(f"Error loading abnormal video {abnormal_path}: {e}")
            bag_abnormal = torch.zeros(
                self.num_segments, 3, self.num_frames, self.img_size, self.img_size
            )
        
        return bag_normal, bag_abnormal

class PrecomputedMILDataset(Dataset):
    """
    Dataset that loads precomputed X3D features with feature-level augmentation.
    Much faster for training as features are already extracted.
    
    Supports:
    - Single view (standard features)
    - Multi-view (multiple augmented versions per video)
    - Feature-level augmentation during loading
    - PCA dimensionality reduction
    """
    
    def __init__(
        self,
        normal_features_dir: str,
        abnormal_features_dir: str,
        feature_augmentation: Optional[Dict] = None,
        multi_view: bool = False,
        num_views: int = 5,
        pca_file: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            normal_features_dir: Directory with normal feature .npy files
            abnormal_features_dir: Directory with abnormal feature .npy files
            feature_augmentation: Dict with augmentation config:
                {
                    'noise_std': 0.05,
                    'dropout_rate': 0.1,
                    'temporal_shift': 2,
                    'temporal_dropout': 0.2,
                    'enable': True
                }
            multi_view: If True, expects multi-view features (video_name_v0.npy, etc.)
            num_views: Number of views per video (if multi_view=True)
            pca_file: Path to PCA transform file (pickle with 'pca' and 'scaler' keys)
            seed: Random seed
        """
        self.normal_dir = Path(normal_features_dir)
        self.abnormal_dir = Path(abnormal_features_dir)
        self.multi_view = multi_view
        self.num_views = num_views
        self.pca_file = pca_file
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load PCA transform if provided
        self.pca = None
        self.scaler = None
        if pca_file is not None and os.path.exists(pca_file):
            with open(pca_file, 'rb') as f:
                pca_data = pickle.load(f)
                self.pca = pca_data['pca']
                self.scaler = pca_data['scaler']
                print(f"Loaded PCA from {pca_file}")
                print(f"  Original dim: {self.pca.n_features_in_}")
                print(f"  Reduced dim: {self.pca.n_components_}")
                print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Setup feature augmentation
        self.feature_augmentation = feature_augmentation or {
            'noise_std': 0.05,
            'dropout_rate': 0.1,
            'temporal_shift': 2,
            'temporal_dropout': 0.2,
            'enable': True
        }
        
        # Load feature paths
        if self.multi_view:
            self.normal_groups = self._load_multi_view_groups(self.normal_dir)
            self.abnormal_groups = self._load_multi_view_groups(self.abnormal_dir)
        else:
            self.normal_features = sorted(list(self.normal_dir.glob("*.npy")))
            self.abnormal_features = sorted(list(self.abnormal_dir.glob("*.npy")))
        
        if self.multi_view:
            if len(self.normal_groups) == 0:
                raise ValueError(f"No multi-view features found in {normal_features_dir}")
            if len(self.abnormal_groups) == 0:
                raise ValueError(f"No multi-view features found in {abnormal_features_dir}")
            
            print(f"Loaded {len(self.normal_groups)} normal video groups (multi-view)")
            print(f"Loaded {len(self.abnormal_groups)} abnormal video groups (multi-view)")
            print(f"Views per video: {num_views}")
        else:
            if len(self.normal_features) == 0:
                raise ValueError(f"No features found in {normal_features_dir}")
            if len(self.abnormal_features) == 0:
                raise ValueError(f"No features found in {abnormal_features_dir}")
            
            print(f"Loaded {len(self.normal_features)} normal feature files")
            print(f"Loaded {len(self.abnormal_features)} abnormal feature files")
        
        if self.feature_augmentation.get('enable', False):
            print(f"Feature augmentation enabled:")
            print(f"  - Noise std: {self.feature_augmentation.get('noise_std', 0.05)}")
            print(f"  - Dropout rate: {self.feature_augmentation.get('dropout_rate', 0.1)}")
            print(f"  - Temporal shift: {self.feature_augmentation.get('temporal_shift', 2)}")
    
    def _load_multi_view_groups(self, directory: Path) -> Dict[str, List[Path]]:
        """Load multi-view features grouped by video name."""
        groups = {}
        npy_files = sorted(list(directory.glob("*.npy")))
        
        for npy_file in npy_files:
            stem = npy_file.stem
            
            if '_v' in stem:
                base_name = stem.rsplit('_v', 1)[0]
            else:
                base_name = stem
            
            if base_name not in groups:
                groups[base_name] = []
            
            groups[base_name].append(npy_file)
        
        for base_name in groups:
            groups[base_name].sort()
        
        return groups
    
    def _apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA transformation to features.
        
        Args:
            features: Feature tensor of shape (num_segments, feature_dim)
            
        Returns:
            Transformed features of shape (num_segments, pca_dim)
        """
        if self.pca is None:
            return features
        
        # Convert to numpy
        features_np = features.numpy()
        original_shape = features_np.shape
        
        # Reshape to 2D for PCA
        features_2d = features_np.reshape(-1, features_np.shape[-1])
        
        # Apply scaling and PCA
        features_scaled = self.scaler.transform(features_2d)
        features_pca = self.pca.transform(features_scaled)
        
        # Reshape back
        new_shape = list(original_shape)
        new_shape[-1] = features_pca.shape[-1]
        features_pca = features_pca.reshape(new_shape)
        
        return torch.from_numpy(features_pca.astype(np.float32))
    
    def _apply_feature_augmentation(self, features: torch.Tensor,
                                   is_training: bool = True) -> torch.Tensor:
        """Apply feature-level augmentations."""
        if not is_training or not self.feature_augmentation.get('enable', False):
            return features
        
        # Add Gaussian noise
        if self.feature_augmentation.get('noise_std', 0) > 0:
            features = FeatureAugmentation.add_gaussian_noise(
                features,
                noise_std=self.feature_augmentation['noise_std']
            )
        
        # Apply feature dropout
        if self.feature_augmentation.get('dropout_rate', 0) > 0:
            features = FeatureAugmentation.feature_dropout(
                features,
                dropout_rate=self.feature_augmentation['dropout_rate']
            )
        
        # Apply temporal shift
        if self.feature_augmentation.get('temporal_shift', 0) > 0:
            features = FeatureAugmentation.temporal_shuffle(
                features,
                max_shift=self.feature_augmentation['temporal_shift']
            )
        
        # Apply temporal dropout
        if self.feature_augmentation.get('temporal_dropout', 0) > 0:
            features = FeatureAugmentation.temporal_dropout(
                features,
                dropout_rate=self.feature_augmentation['temporal_dropout']
            )
        
        return features
    
    def __len__(self):
        if self.multi_view:
            return len(self.abnormal_groups)
        else:
            return len(self.abnormal_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get paired features."""
        if self.multi_view:
            abnormal_base_names = list(self.abnormal_groups.keys())
            abnormal_base = abnormal_base_names[idx]
            abnormal_paths = self.abnormal_groups[abnormal_base]
            abnormal_path = random.choice(abnormal_paths)
            
            normal_base_names = list(self.normal_groups.keys())
            normal_base = random.choice(normal_base_names)
            normal_paths = self.normal_groups[normal_base]
            normal_path = random.choice(normal_paths)
        else:
            abnormal_path = self.abnormal_features[idx]
            normal_idx = random.randint(0, len(self.normal_features) - 1)
            normal_path = self.normal_features[normal_idx]
        
        # Load features
        features_normal = torch.from_numpy(np.load(normal_path)).float()
        features_abnormal = torch.from_numpy(np.load(abnormal_path)).float()
        
        # Apply PCA if available
        features_normal = self._apply_pca(features_normal)
        features_abnormal = self._apply_pca(features_abnormal)
        
        # Apply feature augmentation
        features_normal = self._apply_feature_augmentation(features_normal, is_training=True)
        features_abnormal = self._apply_feature_augmentation(features_abnormal, is_training=True)
        
        return features_normal, features_abnormal


def extract_x3d_features_hook(model, device):
    """
    Setup hook to extract 2048-dim features from X3D before final classification.
    
    Args:
        model: X3D model
        device: Torch device
        
    Returns:
        Hook handle and feature container
    """
    features_container = {}
    
    def hook_fn(module, input, output):
        # Extract features after pool but before final projection
        # Output shape: (B, 2048, 1, 1, 1)
        features_container['features'] = output.squeeze()
    
    # Find the output_pool layer (AdaptiveAvgPool3d before final linear)
    hook_handle = None
    for name, module in model.named_modules():
        if 'output_pool' in name or (isinstance(module, torch.nn.AdaptiveAvgPool3d)):
            print(f"Registering hook at: {name}")
            hook_handle = module.register_forward_hook(hook_fn)
            break
    
    if hook_handle is None:
        print("Warning: Could not find output_pool layer, trying alternative...")
        # Try to find the last pooling layer before classification
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.AdaptiveAvgPool3d):
                print(f"Registering hook at: {name}")
                hook_handle = module.register_forward_hook(hook_fn)
    
    return hook_handle, features_container


# Add this to the existing file

def precompute_x3d_features(
    dataset: PairedMILVideoDataset,
    x3d_model,
    output_dir: str,
    device: torch.device,
    batch_size: int = 8
):
    """
    Precompute 432-dim X3D features for entire dataset.
    
    Args:
        dataset: PairedMILVideoDataset instance
        x3d_model: Loaded X3D model (X3DModel instance)
        output_dir: Directory to save features
        device: Torch device
        batch_size: Batch size for processing
    """
    output_path = Path(output_dir)
    normal_dir = output_path / "normal"
    abnormal_dir = output_path / "abnormal"
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir.mkdir(parents=True, exist_ok=True)
    
    x3d_model.eval()
    x3d_model.to(device)
    
    print(f"Precomputing 432-dim features for {len(dataset)} video pairs...")
    
    # Process normal videos
    print("\nProcessing normal videos...")
    normal_processed = 0
    
    for video_path in tqdm(dataset.normal_videos, desc="Normal videos"):
        try:
            # Load video bag (no augmentation)
            bag = dataset._load_video_bag(video_path, apply_augmentation=False)
            s, c, t, h, w = bag.shape
            
            # Extract features for each segment
            segment_features = []
            for seg_idx in range(s):
                segment = bag[seg_idx:seg_idx+1].to(device)  # (1, C, T, H, W)
                
                with torch.no_grad():
                    feat = x3d_model.extract_features(segment)  # (1, 432)
                
                segment_features.append(feat.cpu().squeeze(0))
            
            # Stack: (S, 432)
            features = torch.stack(segment_features).numpy()
            
            # Save
            base_name = Path(video_path).stem
            np.save(normal_dir / f"{base_name}.npy", features)
            normal_processed += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # Process abnormal videos
    print("\nProcessing abnormal videos...")
    abnormal_processed = 0
    
    for video_path in tqdm(dataset.abnormal_videos, desc="Abnormal videos"):
        try:
            bag = dataset._load_video_bag(video_path, apply_augmentation=False)
            s, c, t, h, w = bag.shape
            
            segment_features = []
            for seg_idx in range(s):
                segment = bag[seg_idx:seg_idx+1].to(device)
                
                with torch.no_grad():
                    feat = x3d_model.extract_features(segment)
                
                segment_features.append(feat.cpu().squeeze(0))
            
            features = torch.stack(segment_features).numpy()
            
            base_name = Path(video_path).stem
            np.save(abnormal_dir / f"{base_name}.npy", features)
            abnormal_processed += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    print(f"\nFeature extraction complete!")
    print(f"  Normal: {normal_processed}/{len(dataset.normal_videos)} files")
    print(f"  Abnormal: {abnormal_processed}/{len(dataset.abnormal_videos)} files")
    print(f"  Feature dimension: 432")
    print(f"Features saved to {output_dir}")
    

def compute_pca_transform(
    features_dir: str,
    output_file: str,
    n_components: int = 512,
    max_samples: Optional[int] = None
):
    """
    Compute PCA transformation from precomputed features.
    
    Args:
        features_dir: Directory with .npy feature files
        output_file: Path to save PCA pickle file
        n_components: Number of PCA components
        max_samples: Maximum samples to use (None = all)
    """
    features_dir = Path(features_dir)
    
    print(f"Computing PCA with {n_components} components...")
    print(f"Loading features from {features_dir}")
    
    # Load all features
    all_features = []
    feature_files = list(features_dir.glob("*.npy"))
    
    if max_samples is not None:
        feature_files = feature_files[:max_samples]
    
    for feat_file in tqdm(feature_files, desc="Loading features"):
        features = np.load(feat_file)  # (S, 2048)
        all_features.append(features)
    
    # Stack: (N_videos * S, 2048)
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"Loaded {all_features.shape[0]} feature vectors of dim {all_features.shape[1]}")
    
    # Fit scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # Fit PCA
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)
    
    # Save
    pca_data = {
        'pca': pca,
        'scaler': scaler,
        'n_components': n_components,
        'original_dim': all_features.shape[1],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(pca_data, f)
    
    print(f"\nPCA transform saved to {output_file}")
    print(f"  Original dim: {all_features.shape[1]}")
    print(f"  Reduced dim: {n_components}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Cumulative variance (first 10): {pca.explained_variance_ratio_[:10].sum():.4f}")


def analyze_pca_components(
    features_dir: str,
    output_dir: str,
    component_sizes: List[int] = [64, 128, 256, 512, 1024],
    max_samples: Optional[int] = None
):
    """
    Analyze PCA with different component sizes and generate report.
    
    Args:
        features_dir: Directory with .npy feature files
        output_dir: Directory to save PCA files and report
        component_sizes: List of PCA component sizes to test
        max_samples: Maximum samples to use (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PCA Analysis Report")
    print("="*80)
    
    # Load features
    features_dir = Path(features_dir)
    all_features = []
    feature_files = list(features_dir.glob("*.npy"))
    
    if max_samples is not None:
        feature_files = feature_files[:max_samples]
    
    print(f"\nLoading features from {len(feature_files)} files...")
    for feat_file in tqdm(feature_files, desc="Loading"):
        features = np.load(feat_file)
        all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"Total feature vectors: {all_features.shape[0]}")
    print(f"Original dimensionality: {all_features.shape[1]}")
    
    # Fit scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # Store results
    results = []
    
    # Test each component size
    for n_comp in component_sizes:
        if n_comp > all_features.shape[1]:
            print(f"\nSkipping {n_comp} (larger than original dim {all_features.shape[1]})")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing PCA with {n_comp} components")
        print(f"{'='*60}")
        
        # Fit PCA
        pca = PCA(n_components=n_comp)
        features_pca = pca.fit_transform(features_scaled)
        
        # Compute metrics
        explained_var = pca.explained_variance_ratio_.sum()
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Reconstruction error
        features_reconstructed = pca.inverse_transform(features_pca)
        mse = np.mean((features_scaled - features_reconstructed) ** 2)
        
        print(f"  Explained variance: {explained_var:.6f}")
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Compression ratio: {all_features.shape[1] / n_comp:.2f}x")
        
        # Save PCA transform
        pca_file = output_dir / f"pca_{n_comp}.pkl"
        pca_data = {
            'pca': pca,
            'scaler': scaler,
            'n_components': n_comp,
            'original_dim': all_features.shape[1],
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_var,
            'reconstruction_mse': mse
        }
        
        with open(pca_file, 'wb') as f:
            pickle.dump(pca_data, f)
        
        print(f"  Saved to: {pca_file}")
        
        results.append({
            'n_components': n_comp,
            'explained_variance': explained_var,
            'reconstruction_mse': mse,
            'compression_ratio': all_features.shape[1] / n_comp,
            'cumulative_variance': cumulative_var
        })
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Explained variance vs components
    n_comps = [r['n_components'] for r in results]
    exp_vars = [r['explained_variance'] for r in results]
    
    axes[0, 0].plot(n_comps, exp_vars, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Components', fontsize=12)
    axes[0, 0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0, 0].set_title('Explained Variance vs PCA Components', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Plot 2: Reconstruction error
    mses = [r['reconstruction_mse'] for r in results]
    
    axes[0, 1].plot(n_comps, mses, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Number of Components', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction MSE', fontsize=12)
    axes[0, 1].set_title('Reconstruction Error vs PCA Components', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Compression ratio
    comp_ratios = [r['compression_ratio'] for r in results]
    
    axes[1, 0].bar(range(len(n_comps)), comp_ratios, color='green', alpha=0.7)
    axes[1, 0].set_xticks(range(len(n_comps)))
    axes[1, 0].set_xticklabels(n_comps)
    axes[1, 0].set_xlabel('Number of Components', fontsize=12)
    axes[1, 0].set_ylabel('Compression Ratio', fontsize=12)
    axes[1, 0].set_title('Compression Ratio vs PCA Components', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative variance (for largest PCA)
    if results:
        largest_pca_result = results[-1]
        cum_var = largest_pca_result['cumulative_variance']
        
        axes[1, 1].plot(range(1, len(cum_var) + 1), cum_var, linewidth=2)
        axes[1, 1].set_xlabel('Number of Components', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1, 1].set_title(f'Cumulative Variance (PCA-{largest_pca_result["n_components"]})', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[1, 1].axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plot_file = output_dir / "pca_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    # Generate text report
    report_file = output_dir / "pca_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PCA Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total feature vectors: {all_features.shape[0]}\n")
        f.write(f"  Original dimensionality: {all_features.shape[1]}\n")
        f.write(f"  Number of files analyzed: {len(feature_files)}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"PCA Component Analysis\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"PCA with {result['n_components']} components:\n")
            f.write(f"  Explained variance: {result['explained_variance']:.6f}\n")
            f.write(f"  Reconstruction MSE: {result['reconstruction_mse']:.6f}\n")
            f.write(f"  Compression ratio: {result['compression_ratio']:.2f}x\n")
            f.write(f"  File: pca_{result['n_components']}.pkl\n")
            f.write("\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"Recommendations:\n")
        f.write(f"{'='*80}\n\n")
        
        # Find best tradeoff
        for result in results:
            if result['explained_variance'] >= 0.95:
                f.write(f"✓ For 95% variance: Use PCA-{result['n_components']} ")
                f.write(f"(MSE: {result['reconstruction_mse']:.6f}, ")
                f.write(f"{result['compression_ratio']:.2f}x compression)\n")
                break
        
        for result in results:
            if result['explained_variance'] >= 0.99:
                f.write(f"✓ For 99% variance: Use PCA-{result['n_components']} ")
                f.write(f"(MSE: {result['reconstruction_mse']:.6f}, ")
                f.write(f"{result['compression_ratio']:.2f}x compression)\n")
                break
        
        f.write(f"\nBalanced recommendation (variance vs compression):\n")
        # Find best balance (maximize explained variance / compression ratio)
        best_balance = max(results, key=lambda r: r['explained_variance'] / (r['compression_ratio'] / 10))
        f.write(f"  PCA-{best_balance['n_components']} provides good balance\n")
        f.write(f"  Variance: {best_balance['explained_variance']:.4f}, ")
        f.write(f"Compression: {best_balance['compression_ratio']:.2f}x\n")
    
    print(f"Report saved to: {report_file}")
    
    print(f"\n{'='*80}")
    print("PCA Analysis Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - PCA transforms: pca_{{64,128,256,512,1024}}.pkl")
    print(f"  - Analysis plots: pca_analysis.png")
    print(f"  - Text report: pca_report.txt")

# Test and usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from model import get_model
    import yaml
    
    print("="*80)
    print("Testing PairedMILVideoDataset")
    print("="*80)
    
    # Test dataset
    dataset = PairedMILVideoDataset(
        normal_dir="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/nonviolence",
        abnormal_dir="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/violence",
        num_frames=16,
        frame_step=2,
        num_segments=5,
        img_size=224,
        seed=42
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading
    print("\nTesting sample loading...")
    bag_normal, bag_abnormal = dataset[0]
    print(f"Normal bag shape: {bag_normal.shape}")  # Expected: (5, 3, 16, 224, 224)
    print(f"Abnormal bag shape: {bag_abnormal.shape}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    for batch_idx, (normal_bags, abnormal_bags) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Normal: {normal_bags.shape}")  # (B, S, C, T, H, W)
        print(f"  Abnormal: {abnormal_bags.shape}")
        if batch_idx >= 2:
            break
    
    # Test X3D feature extraction
    print("\n" + "="*80)
    print("Testing X3D Feature Extraction")
    print("="*80)
    
    # Load X3D model
    config_path = "/home/atin-ct3/action_recognition/config/X3D/rfw.yaml"
    checkpoint_path = "/home/atin-ct3/action_recognition/checkpoints/x3d_rfw_violence/best_model_acc.pth"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading X3D model from {checkpoint_path}...")
    model = get_model(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test feature extraction on single pair
    print("\nExtracting features for single pair...")
    bag_normal, bag_abnormal = dataset[0]
    
    # Combine and flatten
    combined = torch.stack([bag_normal, bag_abnormal])  # (2, 5, 3, 16, 224, 224)
    b, s, c, t, h, w = combined.shape
    combined_flat = combined.view(b * s, c, t, h, w).to(device)  # (10, 3, 16, 224, 224)
    
    print(f"Input shape to X3D: {combined_flat.shape}")
    
    with torch.no_grad():
        features = model(combined_flat)
    
    print(f"Output features shape: {features.shape}")  # (10, 2048)
    
    # Reshape back
    features_reshaped = features.view(b, s, -1)  # (2, 5, 2048)
    print(f"Reshaped features: {features_reshaped.shape}")
    print(f"  Normal features: {features_reshaped[0].shape}")
    print(f"  Abnormal features: {features_reshaped[1].shape}")
    
    # # Uncomment to precompute all features
    # print("\n" + "="*80)
    # print("Precomputing Features (Commented out - uncomment to run)")
    # print("="*80)
    # precompute_x3d_features(
    #     dataset=dataset,
    #     x3d_model=model,
    #     output_dir="/home/atin-ct3/action_recognition/data/mil_features",
    #     device=device,
    #     batch_size=8
    # )
    
    # Test precomputed dataset
    print("\n" + "="*80)
    print("Testing PrecomputedMILDataset (requires precomputed features)")
    print("="*80)
    
    # This will fail if features not precomputed - that's expected
    try:
        precomputed_dataset = PrecomputedMILDataset(
            normal_features_dir="/home/atin-ct3/action_recognition/data/mil_features/normal",
            abnormal_features_dir="/home/atin-ct3/action_recognition/data/mil_features/abnormal",
            seed=42
        )
        
        print(f"Precomputed dataset length: {len(precomputed_dataset)}")
        
        features_normal, features_abnormal = precomputed_dataset[0]
        print(f"Normal features shape: {features_normal.shape}")
        print(f"Abnormal features shape: {features_abnormal.shape}")
    except Exception as e:
        print(f"Precomputed dataset not available (expected): {e}")
        print("Run precompute_x3d_features() first to generate features.")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)