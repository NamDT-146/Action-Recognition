#!/bin/bash

# Training script for MIL adapters

# Configuration
NORMAL_FEATURES="data/RFW-2000-cleaned/mil_features/normal"
ABNORMAL_FEATURES="data/RFW-2000-cleaned/mil_features/abnormal"
OUTPUT_DIR="checkpoints/mil_adapter"

# Training parameters
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=0.001

# Train MLP adapter
# echo "Training MLP adapter..."
# uv run python scripts/train_mil_adapter.py \
#     --normal_features_dir $NORMAL_FEATURES \
#     --abnormal_features_dir $ABNORMAL_FEATURES \
#     --adapter_type mlp \
#     --hidden_dim 32 \
#     --dropout 0.6 \
#     --loss_type ranking \
#     --margin 1.0 \
#     --use_augmentation \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LEARNING_RATE \
#     --optimizer adam \
#     --scheduler plateau \
#     --output_dir $OUTPUT_DIR

# Train LSTM adapter
echo "Training LSTM adapter..."
uv run python scripts/train_mil_adapter.py \
    --normal_features_dir $NORMAL_FEATURES \
    --abnormal_features_dir $ABNORMAL_FEATURES \
    --adapter_type lstm \
    --hidden_dim 32 \
    --dropout 0.6 \
    --loss_type ranking_sparsity \
    --margin 1.0 \
    --sparsity_weight 0.01 \
    --use_augmentation \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --optimizer adam \
    --scheduler plateau \
    --output_dir $OUTPUT_DIR

# Train Conv1D adapter
# echo "Training Conv1D adapter..."
# uv run python scripts/train_mil_adapter.py \
#     --normal_features_dir $NORMAL_FEATURES \
#     --abnormal_features_dir $ABNORMAL_FEATURES \
#     --adapter_type conv1d \
#     --hidden_dim 32 \
#     --dropout 0.6 \
#     --loss_type ranking_smoothing \
#     --margin 1.0 \
#     --smoothing_weight 0.01 \
#     --use_augmentation \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LEARNING_RATE \
#     --optimizer adam \
#     --scheduler plateau \
#     --output_dir $OUTPUT_DIR

echo "Training complete!"