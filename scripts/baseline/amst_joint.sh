#!/bin/bash

# AMST Joint Training Script
# Based on train_amst_joint.py framework

# Basic settings
LR=0.001
LR_VISUAL=0.001
LR_AUDIO=0.001
LR_TEXT=0.00001
LR_IMAGE=0.001
LR_FUSION=0.001
BS=64
EPOCH=100
SEED=42
GPU_IDS="0"

# Choose dataset and configure skip factors
# Uncomment the dataset you want to use

# # ========== CREMAD (Audio-Visual, 6 classes) ==========
# DATASET="CREMAD"
# MODEL_NAME='["Visual","Audio"]'
# SKIP_VISUAL=1
# SKIP_AUDIO=5
# SKIP_TEXT=0
# SKIP_IMAGE=0
# NUM_CLASSES=6

# ========== AVE (Audio-Visual, 28 classes) ==========
DATASET="AVE"
MODEL_NAME='["Visual","Audio"]'
SKIP_VISUAL=1
SKIP_AUDIO=2
SKIP_TEXT=0
SKIP_IMAGE=0
NUM_CLASSES=28

# ========== KineticSound (Audio-Visual, 34 classes) ==========
# DATASET="KineticSound"
# MODEL_NAME='["Visual","Audio"]'
# SKIP_VISUAL=1
# SKIP_AUDIO=3
# SKIP_TEXT=0
# SKIP_IMAGE=0
# NUM_CLASSES=34

# ========== IEMOCAP3 (Text-Visual-Audio, 5 classes) ==========
# DATASET="IEMOCAP3"
# MODEL_NAME='["Text","Visual","Audio"]'
# SKIP_VISUAL=1
# SKIP_AUDIO=4
# SKIP_TEXT=10
# SKIP_IMAGE=0
# NUM_CLASSES=5

# ========== MVSA (Image-Text, 3 classes) ==========
# DATASET="MVSA"
# MODEL_NAME='["Image","Text"]'
# SKIP_VISUAL=0
# SKIP_AUDIO=0
# SKIP_TEXT=10
# SKIP_IMAGE=1
# NUM_CLASSES=3

# ========== Food101 (Image-Text, 101 classes) ==========
# DATASET="Food101"
# MODEL_NAME='["Image","Text"]'
# SKIP_VISUAL=0
# SKIP_AUDIO=0
# SKIP_TEXT=10
# SKIP_IMAGE=1
# NUM_CLASSES=101

# Model settings
FUSION_METHOD="concat"  # AMST Joint only supports concat
UNIFIED_DIM=512
M1_TOKEN_LEN=1
M2_TOKEN_LEN=1

# AME settings
MASK_TYPE="None"  # or "AME"
ALPHA=2.0
AME_GAMA=0.1
AME_BETA=0.7
AME_TEMPERATURE=0.2
WARMUP_EPOCH=0
USE_MACE=True

# Optimizer settings
OPTIMIZER="Adamw"  # SGD, Adam, Adamw
LR_DECAY_STEP=30
LR_DECAY_RATIO=0.5
WEIGHT_DECAY=0.0005

# Paths
CKPT_PATH="./checkpoint"
MODEL_SAVE_NAME="amst_joint_${DATASET}_skip_a${SKIP_AUDIO}_v${SKIP_VISUAL}_t${SKIP_TEXT}"

# Tensorboard (optional)
USE_TENSORBOARD=False
# TENSORBOARD_PATH="./tensorboard/${MODEL_SAVE_NAME}"

# Create checkpoint directory
mkdir -p ${CKPT_PATH}

# echo "=========================================="
# echo "AMST Joint Training Configuration"
# echo "=========================================="
# echo "Dataset: ${DATASET}"
# echo "Model Name: ${MODEL_NAME}"
# echo "Batch Size: ${BS}"
# echo "Epochs: ${EPOCH}"
# echo "Learning Rate (Visual): ${LR_VISUAL}"
# echo "Learning Rate (Audio): ${LR_AUDIO}"
# echo "Learning Rate (Text): ${LR_TEXT}"
# echo "Learning Rate (Image): ${LR_IMAGE}"
# echo "Learning Rate (Fusion): ${LR_FUSION}"
# echo "Skip Factor (Visual): ${SKIP_VISUAL}"
# echo "Skip Factor (Audio): ${SKIP_AUDIO}"
# echo "Skip Factor (Text): ${SKIP_TEXT}"
# echo "Skip Factor (Image): ${SKIP_IMAGE}"
# echo "Fusion Method: ${FUSION_METHOD}"
# echo "Random Seed: ${SEED}"
# echo "GPU IDs: ${GPU_IDS}"
# echo "Save Path: ${CKPT_PATH}/${MODEL_SAVE_NAME}"
# echo "=========================================="
# echo ""

# Run training
python train_amst_joint.py \
    --dataset ${DATASET} \
    --model_name "${MODEL_NAME}" \
    --batch_size ${BS} \
    --epochs ${EPOCH} \
    --train \
    --num_classes ${NUM_CLASSES} \
    --learning_rate_visual ${LR_VISUAL} \
    --learning_rate_audio ${LR_AUDIO} \
    --learning_rate_text ${LR_TEXT} \
    --learning_rate_image ${LR_IMAGE} \
    --learning_rate_fusion ${LR_FUSION} \
    --optimizer ${OPTIMIZER} \
    --lr_decay_step ${LR_DECAY_STEP} \
    --lr_decay_ratio ${LR_DECAY_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --skip_visual ${SKIP_VISUAL} \
    --skip_audio ${SKIP_AUDIO} \
    --skip_text ${SKIP_TEXT} \
    --skip_image ${SKIP_IMAGE} \
    --fusion_method ${FUSION_METHOD} \
    --unified_dim ${UNIFIED_DIM} \
    --m1_token_len ${M1_TOKEN_LEN} \
    --m2_token_len ${M2_TOKEN_LEN} \
    --MaskType ${MASK_TYPE} \
    --alpha ${ALPHA} \
    --ame_gama ${AME_GAMA} \
    --ame_beta ${AME_BETA} \
    --ame_temperature ${AME_TEMPERATURE} \
    --warmup_epoch ${WARMUP_EPOCH} \
    --Use_MACE ${USE_MACE} \
    --random_seed ${SEED} \
    --gpu_ids ${GPU_IDS} \
    --ckpt_path ${CKPT_PATH} \
    --model_save_name ${MODEL_SAVE_NAME} \
    --fps 2 \

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${CKPT_PATH}/${MODEL_SAVE_NAME}"
echo "=========================================="
