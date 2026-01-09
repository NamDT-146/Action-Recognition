import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune TimeSformer on action recognition')
    parser.add_argument('--config', type=str, 
                       default='config/TimeSformer/human7action.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TimeSformer Fine-tuning for Action Recognition")
    print("=" * 80)
    
    main(args.config)