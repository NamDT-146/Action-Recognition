# Example usage of load_pretrained_model for loading a pretrained 3D ResNet model

import torch
from ResNet3D.model import generate_model, load_pretrained_model

model_depth = 34  # Example depth, can be 18, 34, 50, etc.

# Define options as an object with required attributes
class Opt:
    model = 'resnet'
    model_depth = model_depth
    n_classes = 700  # Number of classes in pretraining
    n_input_channels = 3  # Number of input channels (e.g., RGB)
    resnet_shortcut = 'B'
    conv1_t_size = 7
    conv1_t_stride = 1
    no_max_pool = False
    resnet_widen_factor = 1.0

opt = Opt()



# Generate the model
model = generate_model(opt)

print(model)

# Load the pretrained weights and set the final layer for fine-tuning
pretrain_path = 'r3d34_K_200ep.pth'
model_name = 'resnet'
n_finetune_classes = 700  # Set to your target number of classes



model = load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes)

print(model)

import torch.nn as nn

# Replace the last fully connected layer with a binary classifier (2 output nodes + softmax)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 2),
    nn.Softmax(dim=1)
)

print(model)

from optflowdataset import create_data_loaders

# Data parameters
data_dir = "data"
dataset_name = "testdataset"  # Path: data/testdataset with violence and nonviolence subfolders
batch_size = 16
frame_size = 128
num_frames = 16

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    data_dir=data_dir,
    dataset_name=dataset_name,
    batch_size=batch_size,
    frame_size=frame_size,
    num_frames=num_frames,
    train_split=0.7,
    val_split=0.1,
    num_workers=0,  # Set to 0 to avoid CUDA initialization issues
    split_file=f"data/{dataset_name}_split.csv"  # Save splits for reproducibility
)

# Display dataset information
print(f"Dataset loaded with frame size {frame_size}x{frame_size}, {num_frames} frames per video")
print(f"Batch size: {batch_size}")

# Verify input shape matches model expectations
for batch_idx, (sequences, labels) in enumerate(train_loader):
    print(f"Input shape: {sequences.shape}")  # Should be [batch_size, 3, 150, 512, 512]
    print(f"Labels: {labels}")
    break  # Just check the first batch