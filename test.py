import cv2
import os
from pathlib import Path


def test_video_properties(video_path):
    """
    Test and print video properties.
    
    Args:
        video_path (str): Path to video file
    """
    print(f"\n{'='*60}")
    print(f"Testing video: {video_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ File does not exist: {video_path}")
        return False
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print(f"✓ File exists | Size: {file_size:.2f} MB")
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video file")
        return False
    
    print(f"✓ Video opened successfully")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\nVideo Properties:")
    print(f"  Frames: {frame_count}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Codec: {fourcc}")
    
    # Validate properties
    if width == 0 or height == 0:
        print(f"❌ Invalid resolution: {width}x{height}")
        cap.release()
        return False
    
    if frame_count == 0:
        print(f"❌ No frames detected")
        cap.release()
        return False
    
    # Try to read frames
    print(f"\nReading frames...")
    frame_num = 0
    failed_frames = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame is None or frame.size == 0:
            failed_frames.append(frame_num)
        
        frame_num += 1
        
        if frame_num % max(1, frame_count // 10) == 0:
            print(f"  Read {frame_num}/{frame_count} frames...")
    
    cap.release()
    
    if failed_frames:
        print(f"❌ {len(failed_frames)} frames failed to read: {failed_frames[:10]}")
        return False
    else:
        print(f"✓ All {frame_num} frames read successfully")
        return True


def extract_frames_safely(video_path, num_frames=8, target_size=224):
    """
    Extract frames from video with error handling.
    
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract
        target_size (int): Target frame size (224x224)
        
    Returns:
        tuple: (frames, success) where frames is list of numpy arrays
    """
    print(f"\n{'='*60}")
    print(f"Extracting {num_frames} frames from: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return [], False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Validate
    if width == 0 or height == 0:
        print(f"❌ Invalid resolution: {width}x{height}")
        cap.release()
        return [], False
    
    if frame_count == 0:
        print(f"❌ No frames in video")
        cap.release()
        return [], False
    
    print(f"Video info: {frame_count} frames, {width}x{height}")
    
    # Calculate frame indices to extract
    frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    frames = []
    failed_indices = []
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None or frame.size == 0:
            print(f"  ❌ Failed to read frame {frame_idx}")
            failed_indices.append(frame_idx)
            continue
        
        try:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize with safety check
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                frame = cv2.resize(frame, (target_size, target_size), 
                                   interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                print(f"  ✓ Frame {frame_idx}: {frame.shape}")
            else:
                print(f"  ❌ Frame {frame_idx} has invalid shape: {frame.shape}")
                failed_indices.append(frame_idx)
        
        except Exception as e:
            print(f"  ❌ Error processing frame {frame_idx}: {e}")
            failed_indices.append(frame_idx)
    
    cap.release()
    
    success = len(frames) == num_frames
    print(f"\nExtraction result: {len(frames)}/{num_frames} frames extracted")
    
    if failed_indices:
        print(f"Failed frame indices: {failed_indices}")
    
    return frames, success


# Test specific video
if __name__ == "__main__":
    # Test the problematic video
    video_path = "/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/violence/Fight_67.mp4"
    
    # Step 1: Check video properties
    is_valid = test_video_properties(video_path)
    
    # Step 2: Try to extract frames
    if is_valid:
        frames, success = extract_frames_safely(video_path, num_frames=8, target_size=224)
        
        if success:
            print(f"\n✓ Video processing successful!")
            print(f"  Extracted {len(frames)} frames of shape {frames[0].shape}")
        else:
            print(f"\n❌ Video processing failed!")
    else:
        print(f"\n⚠️  Video validation failed - file may be corrupted")
        print(f"\nTroubleshooting steps:")
        print(f"1. Try to play the video with: ffplay {video_path}")
        print(f"2. Check file integrity: ffprobe {video_path}")
        print(f"3. Re-download or repair the video file")