from dataset.loader.MILDataset import PairedMILVideoDataset, precompute_x3d_features, PrecomputedMILDataset
from model import get_model
import torch
import yaml

# Load dataset
dataset = PairedMILVideoDataset(
    normal_dir="data/RFW-2000-cleaned/nonviolence",
    abnormal_dir="data/RFW-2000-cleaned/violence",
    augmentation=None  # No augmentation for feature extraction
)

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

# Extract features
# precompute_x3d_features(
#     dataset=dataset,
#     x3d_model=model,
#     output_dir="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features",
#     device=device,
#     batch_size=8
# )

precompute_dataset = PrecomputedMILDataset(
    normal_features_dir="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features/normal",
    abnormal_features_dir="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features/abnormal",
    # pca_file="/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features/pca_analysis"  # Use 256-dim PCA
)

# Test dataset
print(f"Number of samples in precomputed dataset: {len(precompute_dataset)}")
normal_sample, abnormal_sample = precompute_dataset[0]
print(f"Normal sample shape: {normal_sample.shape}")
print(f"Abnormal sample shape: {abnormal_sample.shape}")