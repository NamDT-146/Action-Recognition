import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
import logging
from datetime import datetime
import argparse
from pathlib import Path
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model import ViolenceDetectionModel, build_violence_detection_model
from dataset import VideoDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveVideoTester:
    """Video tester with adaptive threshold and sliding window classification"""
    
    def __init__(self, model_path, config):
        """Initialize the adaptive video tester"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Adaptive threshold parameters
        self.initial_threshold = 0.6
        self.min_threshold = 0.5
        self.max_threshold = 0.7
        self.adaptation_rate = 0.1
        
        # Sliding window parameters
        self.window_size = 10
        self.required_positive = 5
        self.segment_length = 20  # Segment length
        self.segment_step = 10    # Step for segmentation
        
        logger.info(f"Initialized with adaptive threshold: {self.initial_threshold}")
        logger.info(f"Threshold range: [{self.min_threshold}, {self.max_threshold}]")
        logger.info(f"Window parameters: {self.required_positive}/{self.window_size} consecutive")
        logger.info(f"Segmentation: {self.segment_length} frames per segment, step {self.segment_step}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        logger.info(f"Loading model from {model_path}")
        
        # Create model architecture
        model, _, _ = build_violence_detection_model(
            seq_len=self.config['seq_length'],
            img_size=self.config['figure_size'],
            cnn_arch=self.config['cnn_arch'],
            pretrained=self.config['pretrained'],
            freeze_cnn=self.config['freeze_cnn'],
            pretrained_coco=self.config['pretrained_coco'],
            temporal_model=self.config['temporal_model'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=1,
            dropout=self.config['dropout'],
            learning_rate=self.config['learning_rate'],
            optimizer_type=self.config['optimizer_type'],
            weight_init=self.config['weight_init']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights")
        
        model.to(self.device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        return model
    
    # def _load_and_preprocess_video(self, video_path):
    #     """Load all video frames and preprocess them"""
    #     cap = cv2.VideoCapture(video_path)
    #     frames = []
        
    #     if not cap.isOpened():
    #         raise ValueError(f"Cannot open video: {video_path}")
        
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
        
    #     logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
    #     # Load all frames
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
            
    #         # Resize to 224x224 and convert BGR to RGB
    #         frame = cv2.resize(frame, (224, 224))
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         # Normalize to [0, 1]
    #         frame = frame.astype(np.float32) / 255.0
            
    #         frames.append(frame)
        
    #     cap.release()
        
    #     frames = np.array(frames)  # Shape: (N, 224, 224, 3)
    #     logger.info(f"Loaded and preprocessed {len(frames)} frames")
        
    #     return frames, total_frames, fps
    
    def _load_video_frames(self, video_path):
        """Load video frames with specified sampling rate and ImageNet normalization"""
        cap = cv2.VideoCapture(video_path)
        sequences = []
        frame_indices = []
        frame_count = 0
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # ImageNet normalization parameters
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame to match model input
            frame = cv2.resize(frame, (self.config['figure_size'], self.config['figure_size']))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization: (pixel - mean) / std
            frame = (frame - imagenet_mean) / imagenet_std
            
            sequence.append(frame)
            
            # Sample every frame_step frames starting from beginning
            if frame_count % self.segment_step == 0:
                sequences.append(np.array(sequence))
                sequence = []  # Reset for next segment
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        # sequences = np.array(sequences)
        print(sequences)
        logger.info(f"Loaded {len(sequences)} frames (sampled every {self.segment_step} frames)")
        logger.info(f"Applied ImageNet normalization: mean={imagenet_mean}, std={imagenet_std}")
        
        return sequences, frame_indices, total_frames, fps

    def _create_sequences(self, frames):
        """Create overlapping sequences for prediction"""
        sequences = []
        seq_start_indices = []
        
        print("Len frames: ", len(frames))
        
        if len(frames) < self.config['seq_length']:
            # If not enough frames, repeat the last frame
            while len(frames) < self.config['seq_length']:
                frames = np.concatenate([frames, frames[-1:]], axis=0)
        
        # Create sequences with overlap
        for i in range(0, len(frames) - self.config['seq_length'] + 1, 1):
            sequence = frames[i:i + self.config['seq_length']]
            
            # Apply crop_dark if specified
            if self.config.get('crop_dark'):
                h, w = sequence.shape[1:3]
                crop_h, crop_w = self.config['crop_dark']
                sequence = sequence[:, crop_h:h-crop_h, crop_w:w-crop_w, :]
            
            # Frames are already normalized with ImageNet stats, just transpose to (seq_len, C, H, W)
            sequence = sequence.transpose(0, 3, 1, 2)  # (seq_len, C, H, W)
            
            sequences.append(sequence)
            seq_start_indices.append(i)
            
        print(len(sequences))
        
        return np.array(sequences), seq_start_indices

    def _update_threshold(self, probability, current_threshold):
        """Update threshold based on current prediction using aggressive update formula"""
        new_threshold = current_threshold * (1 - self.adaptation_rate) + probability * self.adaptation_rate
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        return new_threshold
    
    def _sliding_window_decision(self, predictions):
        """Apply sliding window rule"""
        if len(predictions) < self.window_size:
            return sum(predictions) >= len(predictions) // 2
        
        # Check sliding windows
        for i in range(len(predictions) - self.window_size + 1):
            window = predictions[i:i + self.window_size]
            if sum(window) >= self.required_positive:
                return True
        
        return False
    
    def _make_predictions_from_segments(self, segments):
        """Make predictions from list of segments"""
        predictions = []
        probabilities = []
        thresholds = []
        inference_times = []
        
        current_threshold = self.initial_threshold
        total_inference_time = 0
        
        with torch.no_grad():
            for i, segment in enumerate(segments):
                # Prepare input tensor
                input_tensor = torch.from_numpy(segment).unsqueeze(0).to(self.device).float()
                
                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                # Forward pass
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).cpu().numpy()[0][0]
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time
                
                total_inference_time += inference_time
                inference_times.append(inference_time)
                
                # Make prediction with current threshold
                prediction = int(probability > current_threshold)
                
                # Store results
                predictions.append(prediction)
                probabilities.append(probability)
                thresholds.append(current_threshold)
                
                # Update threshold for next prediction
                current_threshold = self._update_threshold(probability, current_threshold)
                
                logger.debug(f"Segment {i+1}/{len(segments)}: "
                           f"prob={probability:.3f}, threshold={thresholds[-1]:.3f}, "
                           f"pred={prediction}, new_threshold={current_threshold:.3f}")
        
        logger.info(f"Processed {len(segments)} segments in {total_inference_time*1000:.2f} ms")
        
        return predictions, probabilities, thresholds, inference_times, total_inference_time
            
    def test_single_video(self, video_path, true_label=None):
        """Test a single video with frame differences and segmentation"""
        video_name = Path(video_path).stem
        logger.info(f"Testing video: {video_name}")
        
        try:
            # Step 1: Load all video frames and preprocess them
            frames, total_frames, fps = self._load_and_preprocess_video(video_path)
            
            # Step 2: Calculate frame differences (n-1 differences from n frames)
            frame_differences = self._calculate_frame_differences(frames)
            
            # Step 3: Create segments with length 20 and step 10 (overlapping segments)
            segments, segment_indices = self._create_segments(frame_differences)
            
            if len(segments) == 0:
                raise ValueError("No valid segments created from video")
            
            # Step 4: Make predictions from segments
            predictions, probabilities, thresholds, inference_times, total_inference_time = \
                self._make_predictions_from_segments(segments)
                
            probs = np.array(probabilities)
            print('Probabilities: ', probs)
            
            # Step 5: Apply sliding window decision
            final_prediction = self._sliding_window_decision(predictions)
            
            # Calculate statistics
            avg_probability = np.mean(probabilities)
            max_probability = np.max(probabilities)
            min_probability = np.min(probabilities)
            avg_threshold = np.mean(thresholds)
            positive_predictions = sum(predictions)
            
            result = {
                'video_name': video_name,
                'video_path': video_path,
                'true_label': true_label,
                'final_prediction': int(final_prediction),
                'total_frames': total_frames,
                'difference_frames': len(frame_differences),
                'total_segments': len(segments),
                'positive_predictions': positive_predictions,
                'positive_rate': positive_predictions / len(segments),
                'avg_probability': avg_probability,
                'max_probability': max_probability,
                'min_probability': min_probability,
                'initial_threshold': self.initial_threshold,
                'avg_threshold': avg_threshold,
                'final_threshold': thresholds[-1] if thresholds else self.initial_threshold,
                'total_inference_time_ms': total_inference_time * 1000,
                'avg_inference_time_ms': np.mean(inference_times) * 1000,
                'fps_original': fps,
                'fps_inference': len(segments) / total_inference_time if total_inference_time > 0 else 0,
                'window_size': self.window_size,
                'required_positive': self.required_positive,
                'segment_length': self.segment_length,
                'segment_step': self.segment_step,
                'segment_indices': segment_indices,
                'segment_predictions': predictions,
                'segment_probabilities': probs,
                'segment_thresholds': thresholds
            }
            
            logger.info(f"Video {video_name}: Final prediction={final_prediction}, "
                    f"Positive segments={positive_predictions}/{len(segments)}, "
                    f"Avg prob={avg_probability:.3f}, Avg threshold={avg_threshold:.3f}")
            
            return result   
            
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {str(e)}")
            return {
                'video_name': video_name,
                'video_path': video_path,
                'true_label': true_label,
                'final_prediction': -1,
                'error': str(e)
            }

    def _load_and_preprocess_video(self, video_path):
        """Load all video frames and preprocess them (similar to VideoDataLoader.py)"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # ImageNet normalization parameters (same as VideoDataLoader.py)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        # Load all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame (same as VideoDataLoader.py)
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply crop_dark if specified
            if self.config.get('crop_dark'):
                h, w = frame_rgb.shape[:2]
                crop_h, crop_w = self.config['crop_dark']
                frame_rgb = frame_rgb[crop_h:h-crop_h, crop_w:w-crop_w]
            
            # Resize to 224x224 (same as config)
            frame_rgb = cv2.resize(frame_rgb, (self.config['figure_size'], self.config['figure_size']))
            
            # Convert to float32 and normalize to [0, 1]
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            frame_rgb = (frame_rgb - imagenet_mean) / imagenet_std
            
            frames.append(frame_rgb)
        
        cap.release()
        
        frames = np.array(frames)  # Shape: (N, 224, 224, 3)
        logger.info(f"Loaded and preprocessed {len(frames)} frames")
        
        return frames, total_frames, fps

    def _calculate_frame_differences(self, frames):
        """Calculate frame differences (n-1 differences from n frames)"""
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to calculate differences")
        
        differences = []
        
        for i in range(1, len(frames)):
            # Calculate absolute difference between consecutive frames
            diff = np.abs(frames[i] - frames[i-1])
            differences.append(diff)
        
        differences = np.array(differences)  # Shape: (N-1, 224, 224, 3)
        logger.info(f"Calculated {len(differences)} frame differences")
        
        return differences

    def _create_segments(self, frame_differences):
        """Create segments with specified length and step (overlapping segments)"""
        segments = []
        segment_indices = []
        
        if len(frame_differences) < self.segment_length:
            # If not enough frames, repeat the last difference
            while len(frame_differences) < self.segment_length:
                frame_differences = np.concatenate([frame_differences, frame_differences[-1:]], axis=0)
        
        # Create overlapping segments: length=20, step=10
        for i in range(0, len(frame_differences) - self.segment_length + 1, self.segment_step):
            segment = frame_differences[i:i + self.segment_length]
            
            # Transpose to (seq_len, C, H, W) format expected by model
            segment = segment.transpose(0, 3, 1, 2)  # (20, 3, 224, 224)
            
            segments.append(segment)
            segment_indices.append(i)
        
        segments = np.array(segments)  # Shape: (num_segments, 20, 3, 224, 224)
        logger.info(f"Created {len(segments)} segments with length {self.segment_length} and step {self.segment_step}")
        
        return segments, segment_indices
            
    def test_dataset(self, dataset_path, dataset_name, max_videos_per_class=None):
        """Test entire dataset with optional random sampling per class"""
        logger.info(f"Testing dataset: {dataset_name}")
        if max_videos_per_class:
            logger.info(f"Randomly sampling {max_videos_per_class} videos per class")
        
        # Get video paths and labels
        nonviolence_dir = os.path.join(dataset_path, dataset_name.lower(), 'nonviolence')
        violence_dir = os.path.join(dataset_path, dataset_name.lower(), 'violence')
        
        video_paths = []
        labels = []
        
        # Non-violence videos
        if os.path.exists(nonviolence_dir):
            nonviolence_files = [f for f in os.listdir(nonviolence_dir) 
                            if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI'))]
            
            if max_videos_per_class and len(nonviolence_files) > max_videos_per_class:
                import random
                random.seed(42)
                nonviolence_files = random.sample(nonviolence_files, max_videos_per_class)
                logger.info(f"Randomly selected {max_videos_per_class} non-violence videos")
            
            for video_file in nonviolence_files:
                video_paths.append(os.path.join(nonviolence_dir, video_file))
                labels.append(0)
        
        # Violence videos
        if os.path.exists(violence_dir):
            violence_files = [f for f in os.listdir(violence_dir) 
                            if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI'))]
            
            if max_videos_per_class and len(violence_files) > max_videos_per_class:
                import random
                random.seed(42)
                violence_files = random.sample(violence_files, max_videos_per_class)
                logger.info(f"Randomly selected {max_videos_per_class} violence videos")
            
            for video_file in violence_files:
                video_paths.append(os.path.join(violence_dir, video_file))
                labels.append(1)
        
        logger.info(f"Found {len(video_paths)} videos "
               f"({len([l for l in labels if l == 0])} non-violence, "
               f"{len([l for l in labels if l == 1])} violence)")
        
        # Test each video
        results = []
        for i, (video_path, true_label) in enumerate(zip(video_paths, labels)):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {Path(video_path).name}")
            result = self.test_single_video(video_path, true_label)
            results.append(result)
        
        return results

def calculate_metrics(results):
    """Calculate evaluation metrics from results"""
    valid_results = [r for r in results if r['final_prediction'] != -1]
    
    if len(valid_results) == 0:
        logger.warning("No valid predictions to evaluate")
        return {}
    
    y_true = [r['true_label'] for r in valid_results]
    y_pred = [r['final_prediction'] for r in valid_results]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Timing statistics
    inference_times = [r['total_inference_time_ms'] for r in valid_results if 'total_inference_time_ms' in r]
    
    metrics = {
        'total_videos': len(results),
        'valid_predictions': len(valid_results),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0,
        'classification_report': classification_report(y_true, y_pred, target_names=['Non-Violence', 'Violence'])
    }
    
    return metrics

def save_results(results, metrics, output_dir):
    """Save test results and metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare dataframe - exclude large arrays for CSV
    df_data = []
    for result in results:
        row = result.copy()
        # Remove large arrays for CSV
        for key in ['segment_indices', 'segment_predictions', 'segment_thresholds', 'error']:
            if key in row:
                del row[key]
        df_data.append(row)
    
    # Save CSV results
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, f'frame_diff_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Save detailed metrics
    metrics_path = os.path.join(output_dir, f'frame_diff_metrics_{timestamp}.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAME DIFFERENCE VIDEO TESTING RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TESTING PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Frame Preprocessing: Resize to 224x224, Normalize [0,1]\n")
        f.write(f"Frame Differences: Absolute difference between consecutive frames\n")
        f.write(f"Segment Length: 20 frames\n")
        f.write(f"Segment Step: 10 frames\n")
        f.write(f"Adaptive Threshold: [0.5, 0.7], Initial=0.6\n")
        f.write(f"Sliding Window: 3 out of 5 consecutive predictions\n\n")
        
        f.write("CLASSIFICATION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Videos: {metrics['total_videos']}\n")
        f.write(f"Valid Predictions: {metrics['valid_predictions']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        cm = np.array(metrics['confusion_matrix'])
        f.write("           Predicted\n")
        f.write("         Non-V  Violence\n")
        f.write(f"True Non-V   {cm[0,0]:3d}     {cm[0,1]:3d}\n")
        f.write(f"Violence     {cm[1,0]:3d}     {cm[1,1]:3d}\n\n")
        
        f.write("TIMING STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms per video\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 70 + "\n")
        f.write(metrics['classification_report'])
    
    logger.info(f"Detailed metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Videos with Frame Differences and Segmentation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name for logging')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for temporal model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dataset_name', type=str, default='testdataset', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='frame_diff_test_results', help='Output directory')
    parser.add_argument('--video_path', type=str, help='Path to single video file (optional)')
    parser.add_argument('--max_videos_per_class', type=int, default=None, 
                       help='Maximum number of videos to randomly sample per class')
    
    args = parser.parse_args()
    
    # Test configuration (should match training config)
    config = {
        'seq_length': 20,
        'figure_size': 224,  # Changed to 224 for consistency
        'crop_dark': (11, 38),
        'cnn_arch': args.model_name,  # Use model_name from args
        'temporal_model': 'convlstm',
        'hidden_dim': args.hidden_dim,
        'pretrained': True,
        'freeze_cnn': True,
        'pretrained_coco': False,
        'dropout': 0.2,
        'learning_rate': 1e-4,
        'optimizer_type': 'adam',
        'weight_init': 'xavier_uniform'
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = AdaptiveVideoTester(args.model_path, config)
    
    if args.video_path:
        # Test single video
        logger.info(f"Testing single video: {args.video_path}")
        result = tester.test_single_video(args.video_path)
        
        print("\n" + "=" * 60)
        print(f"FRAME DIFFERENCE VIDEO TEST RESULTS")
        print("=" * 60)
        print(f"Video: {result['video_name']}")
        print(f"Final Prediction: {result.get('final_prediction', 'ERROR')}")
        if 'total_segments' in result:
            print(f"Total Frames: {result['total_frames']}")
            print(f"Difference Frames: {result['difference_frames']}")
            print(f"Total Segments: {result['total_segments']}")
            print(f"Positive Segments: {result['positive_predictions']}/{result['total_segments']}")
            print(f"Average Probability: {result['avg_probability']:.3f}")
            print(f"Average Threshold: {result['avg_threshold']:.3f}")
            print(f"Total Inference Time: {result['total_inference_time_ms']:.2f} ms")
        print("=" * 60)
    else:
        # Test dataset
        logger.info(f"Testing dataset: {args.dataset_name}")
        results = tester.test_dataset(args.data_dir, args.dataset_name, args.max_videos_per_class)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Save results
        save_results(results, metrics, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"FRAME DIFFERENCE VIDEO TESTING RESULTS - {args.dataset_name.upper()}")
        print("=" * 70)
        if args.max_videos_per_class:
            print(f"Max Videos Per Class: {args.max_videos_per_class} (randomly sampled)")
        print(f"Total Videos Tested: {metrics['total_videos']}")
        print(f"Valid Predictions: {metrics['valid_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms per video")
        print(f"\nResults saved to: {args.output_dir}")
        print("=" * 70)

if __name__ == "__main__":
    main()