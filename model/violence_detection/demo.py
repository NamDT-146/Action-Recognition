import torch
import numpy as np
import cv2
from flownet.run import estimate, estimate_parallel  # Your optical flow estimator
from model import build_violence_detection_model
from i3d.pytorch_i3d import InceptionI3d
from ResNet3D.models.resnet import generate_model
import os
import torch.nn as nn

def frames_to_tensor(frames):
    arr = np.stack(frames, axis=0)  # [N, H, W, 3]
    arr = arr.astype(np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)  # [N, 3, H, W]
    return torch.from_numpy(arr)

def frame_to_tensor(gray_frame, device='cuda'):
    rgb = np.stack([gray_frame] * 3, axis=-1)
    tensor = torch.FloatTensor(
        np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    ).to(device)
    return tensor

def extract_flow_one_by_one(frames, device='cuda'):
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    optical_flows = []
    
    for i in range(len(gray_frames) - 1):
        prev_frame = gray_frames[i]
        curr_frame = gray_frames[i + 1]
        
        prev_tensor = frame_to_tensor(prev_frame, device)
        curr_tensor = frame_to_tensor(curr_frame, device)
        
        with torch.no_grad():
            flow_tensor = estimate(prev_tensor, curr_tensor)
        
        optical_flows.append(flow_tensor)
    
    batch_flows = torch.stack(optical_flows, dim=0)  # [N-1, 2, H, W]
    return batch_flows.to(device)

def extract_flow_batch(frames, device='cuda', stride=8):
    gray_frames = [ frame_to_tensor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)) for frame in frames]
    frames_tensors = torch.stack(gray_frames)
    return estimate_parallel(frames_tensors[:stride], frames_tensors[1:])

def draw_prediction_on_frames(frames, pred, conf, label_map={0: "Normal", 1: "Violent"}):
    color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # Green for normal, Red for violent
    label = label_map[pred]
    text = f"{label} ({conf:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    out_frames = []
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, text, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
        out_frames.append(frame_bgr)
    return out_frames

def load_3d_convolution_model(model_name, checkpoint_path, mode='rgb', device='cuda'):
    if model_name == 'i3d':
        # Setup I3D model
        if mode == 'flow':
            model = InceptionI3d(400, in_channels=2, dropout_keep_prob=0.5)  # For flow input
        else:
            model = InceptionI3d(400, in_channels=3, dropout_keep_prob=0.5)  # Start with pretrained
        model.replace_logits(2)  # Binary classification
        model.logits = nn.Sequential(
            nn.Dropout3d(0.5),
            model.logits
        )
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, 224  # Return model and input size
    
    elif model_name == 'resnet':
        model = generate_model(50)  # ResNet50
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, 128  # Return model and input size
    
    else:
        raise ValueError(f"Unsupported 3D model: {model_name}")

def run_realtime_inference(video_path, checkpoint_path, output_path, device='cuda', mode='rgb', 
                           model_type='spatial_temporal', model_name='cnn_lstm'):
    if model_type == 'spatial_temporal':
        config = {
            'seq_length': 16,
            'img_size': 256,
            'cnn_arch': 'efficientnet_b0',
            'pretrained': True,
            'freeze_cnn': False,
            'pretrained_coco': False,
            'temporal_model': 'convlstm',
            'hidden_dim': 256,
            'num_classes': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'learning_rate': 1e-4,
            'optimizer_type': 'adam',
            'weight_init': 'xavier_uniform',
        }
        model, _, _ = build_violence_detection_model(
            seq_len=config['seq_length'],
            img_size=config['img_size'],
            cnn_arch=config['cnn_arch'],
            pretrained=config['pretrained'],
            freeze_cnn=config['freeze_cnn'],
            pretrained_coco=config['pretrained_coco'],
            temporal_model=config['temporal_model'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            optimizer_type=config['optimizer_type'],
            weight_init=config['weight_init']
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        img_size = config['img_size']
    
    elif model_type == 'convolution3d':
        model, img_size = load_3d_convolution_model(model_name, checkpoint_path, mode, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    
    if model_type == 'spatial_temporal':
        frames_needed = 9  # For 8 flow frames
        stride = 8  # Process 8 frames at a time
        prev_8_cnn = None
    elif model_type == 'convolution3d':
        if mode == 'flow':
            frames_needed = 9  # Same as spatial-temporal (need 9 frames to get 8 flows)
            prev_8_flows = None  # Store previous 8 flow frames
        else:
            frames_needed = 16  # For RGB, still need 16 frames
        stride = 8  # Process 8 frames at a time
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(img_size)
    height = int(img_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def cnn_preprocess(x):
        # For flow: [N, 2, H, W] -> [N, 3, H, W] if needed
        if x.shape[1] == 2:
            zeros = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
            x = torch.cat([x, zeros], dim=1)
        return x
    
    first_batch = True
    while True:
        while len(frame_buffer) < frames_needed:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)
        
        if len(frame_buffer) < frames_needed:
            break  # End of video
        
        if model_type == 'spatial_temporal':
            # Spatial-temporal model processing
            frames_9 = frame_buffer[:9]  # Use first 9 frames
            
            if mode == 'flow':
                flows_8 = extract_flow_batch(frames_9, device=device)  # [8, 2, H, W]
                flows_8 = cnn_preprocess(flows_8)  # [8, 3, H, W] if needed
                with torch.no_grad():
                    curr_8_cnn = model.cnn_extractor(flows_8.unsqueeze(0)).squeeze(0)  # [8, feat_dim]
            elif mode == 'rgb':
                rgb_8 = frames_to_tensor(frames_9[1:]).to(device)  # [8, 3, H, W]
                with torch.no_grad():
                    curr_8_cnn = model.cnn_extractor(rgb_8.unsqueeze(0)).squeeze(0)  # [8, feat_dim]
            else:
                raise ValueError("mode must be 'flow' or 'rgb'")
            
            if prev_8_cnn is None:
                prev_8_cnn = torch.zeros_like(curr_8_cnn)
            seq_16 = torch.cat([prev_8_cnn, curr_8_cnn], dim=0).unsqueeze(0)  # [1, 16, feat_dim]
            
            with torch.no_grad():
                output = model.temporal_model(seq_16)
                output = model.maxpool(output)
                output = model.flatten(output)
                output = model.classifier(output)
                output = model.output_layer(output)
            
            prev_8_cnn = curr_8_cnn.clone()
            
        elif model_type == 'convolution3d':
            # Process for 3D convolution models
            if mode == 'rgb':
                # Convert frames to tensor [1, 3, T, H, W]
                frames_tensor = frames_to_tensor(frame_buffer).to(device)  # [16, 3, H, W]
                frames_tensor = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, 16, H, W]
                
                with torch.no_grad():
                    output = model(frames_tensor)
                
            elif mode == 'flow':
                # Sliding window approach for flow frames - similar to spatial-temporal model
                
                # Get 8 new flow frames from 9 frames
                flows_8 = extract_flow_one_by_one(frame_buffer, device=device)  # [8, 2, H, W]

                # For the first batch, initialize prev_8_flows with zeros
                if prev_8_flows is None:
                    prev_8_flows = torch.zeros_like(flows_8)
                
                # Combine previous 8 flows with new 8 flows to get 16 flow frames
                flows_16 = torch.cat([prev_8_flows, flows_8], dim=0)  # [16, 2, H, W]
                
                # Store current flows as previous flows for next iteration
                prev_8_flows = flows_8.clone()
                
                # Reshape for 3D convolution model [1, 2, 16, H, W]
                flows_16 = flows_16.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 2, 16, H, W]
                
                with torch.no_grad():
                    output = model(flows_16)
        
        # Get prediction and confidence
        if output.dim() > 2:
            output = output.squeeze(-1)
        if output.shape[-1] > 1:
            conf = torch.softmax(output, dim=1).max().item()
            pred = torch.argmax(output, dim=1).item()
        else:
            conf = torch.sigmoid(output).item()
            pred = int(conf > 0.5)
            
        # Print prediction
        print(f"Prediction: {pred}, Confidence: {conf:.2f}")
        
        # Draw prediction on the frames - we'll only draw on the first 8 frames
        # that we're going to remove from the buffer
        frames_to_draw = frame_buffer[:stride]
        drawed_frames = draw_prediction_on_frames(frames_to_draw, pred, conf)
            
        # Write frames to output video
        for f in drawed_frames:
            out_writer.write(f)
            
        # Remove processed frames based on stride
        frame_buffer = frame_buffer[stride:]  # Keep the remaining frames for next batch
        
        
    cap.release()
    out_writer.release()
    print(f"Output video saved to {output_path}")

if __name__ == '__main__':
    # Example using spatial_temporal model (CNN_LSTM)
    # run_realtime_inference(
    #     'demo2.mp4', 
    #     'weights/flow_lstm_cnn_7803.pth', 
    #     'output_cnn_lstm.mp4', 
    #     device='cuda', 
    #     mode='flow',
    #     model_type='spatial_temporal',
    #     model_name='cnn_lstm'
    # )
    
    # Example using I3D model
    run_realtime_inference(
        'demo.mp4', 
        'weights/flow_i3d_9873.pt', 
        'output_i3d.mp4', 
        device='cuda', 
        mode='flow',
        model_type='convolution3d',
        model_name='i3d'
    )
    
    # Example using ResNet3D model
    # run_realtime_inference(
    #     'demo2.mp4', 
    #     'resnet3d/models/best_model.pt', 
    #     'output_resnet.mp4', 
    #     device='cuda', 
    #     mode='flow',
    #     model_type='convolution3d',
    #     model_name='resnet'
    # )