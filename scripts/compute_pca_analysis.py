"""
Script to compute PCA analysis on precomputed X3D features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.loader.MILDataset import analyze_pca_components

if __name__ == "__main__":
    # Configuration
    FEATURES_DIR = "/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features/normal"
    OUTPUT_DIR = "/home/atin-ct3/action_recognition/data/RFW-2000-cleaned/mil_features/pca_analysis"
    
    # Component sizes to test
    COMPONENT_SIZES = [64, 128, 256, 512, 1024]
    
    # Maximum samples to use (None = all, set to smaller number for faster testing)
    MAX_SAMPLES = None  # Use all available
    
    print("="*80)
    print("PCA Analysis for MIL Features")
    print("="*80)
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Component sizes: {COMPONENT_SIZES}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print("="*80)
    
    # Run analysis
    analyze_pca_components(
        features_dir=FEATURES_DIR,
        output_dir=OUTPUT_DIR,
        component_sizes=COMPONENT_SIZES,
        max_samples=MAX_SAMPLES
    )
    
    print("\nDone!")