#!/bin/bash

# AMST Training Script for Linux
# Asynchronous Multi-modal Skip Training

# ==================== Environment Setup ====================
# Uncomment and modify if you need virtual environment
# source /path/to/your/venv/bin/activate

# ==================== Basic Parameters ====================
LR=0.001
LR_VISUAL=0.001
LR_AUDIO=0.001
LR_TEXT=0.00001
BS=64
EPOCH=10
SEED=42
GPU_ID=0
FPS=1
# ==================== Dataset Configuration ====================
# Skip Factor Configuration Guide:
# CREMAD: A_F=5, V_F=1
# AVE: A_F=2, V_F=1
# IEMOCAP2: A_F=4, V_F=1, T_F=1
# IEMOCAP3: A_F=4, V_F=1, T_F=1
# MVSA: A_F=1, V_F=1, T_F=10
# Food101: A_F=1, V_F=1, T_F=1
# KineticSound: A_F=1, V_F=1

# ==================== AMST Specific Parameters ====================
USE_HELPER=True
HELPER_WEIGHT=0.5
ALPHA=1.0

# ==================== Select Dataset ====================
# Uncomment the dataset you want to use

# # CREMAD (Audio-Visual)
# DATASET="CREMAD"
# A_F=5
# V_F=1
# T_F=1  # Not used for audio-visual datasets
# MODEL_NAME='["Visual","Audio"]'
# LR_DR=0.1 # lr_decay_ratio
# LR_DS=70 # lr_decay_step
# WD=1e-4 # weight_decay
# UD=512 # unified_dim
# FPS=1

# # # AVE (Audio-Visual Event)
# DATASET="AVE"
# A_F=2
# V_F=1
# T_F=1  # Not used
# MODEL_NAME='["Visual","Audio"]'
# FPS=2
# LR_DR=0.5 # lr_decay_ratio
# LR_DS=30 # lr_decay_step
# WD=5e-4 # weight_decay
# UD=512 # unified_dim

# # IEMOCAP3 (Text-Visual-Audio)
DATASET="IEMOCAP3"
A_F=4
V_F=1
T_F=1
MODEL_NAME='["Text","Visual","Audio"]'
LR_DR=0.5 # lr_decay_ratio
LR_DS=20 # lr_decay_step
WD=1e-4 # weight_decay
UD=768 # unified_dim
ALPHA=1.0

# # MVSA (Image-Text)
# DATASET="MVSA"
# A_F=1
# V_F=1
# T_F=1
# MODEL_NAME='["Image","Text"]'
# LR_DR=0.5 # lr_decay_ratio
# LR_DS=20 # lr_decay_step
# WD=5e-4 # weight_decay
# UD=768 # unified_dim

# # Food101 (Image-Text)
# DATASET="Food101"
# A_F=1
# V_F=1
# T_F=1
# MODEL_NAME='["Image","Text"]'

# # KineticSound (Audio-Visual)
# DATASET="KineticSound"
# A_F=1
# V_F=1
# T_F=1
# MODEL_NAME='["Visual","Audio"]'



# ==================== Path Configuration ====================
CKPT_PATH="./checkpoint/Amst/${DATASET}"
MODEL_SAVE_NAME="${DATASET}_amst_af${A_F}_vf${V_F}_tf${T_F}"
TENSORBOARD_PATH="./tensorboard/amst_${DATASET}"

# Create directories if they don't exist
mkdir -p ${CKPT_PATH}
mkdir -p ./Results

# ==================== Training Command ====================
echo "=========================================="
echo "Starting AMST Training"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Skip Factors: Audio=${A_F}, Visual=${V_F}, Text=${T_F}"
echo "Batch Size: ${BS}"
echo "Epochs: ${EPOCH}"
echo "Learning Rate: ${LR}"
echo "Random Seed: ${SEED}"
echo "GPU ID: ${GPU_ID}"
echo "=========================================="

python train_amst_full.py \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --batch_size ${BS} \
    --epochs ${EPOCH} \
    --random_seed ${SEED} \
    --gpu_ids ${GPU_ID} \
    --learning_rate_visual ${LR_VISUAL} \
    --learning_rate_audio ${LR_AUDIO} \
    --learning_rate_text ${LR_TEXT} \
    --learning_rate_fusion ${LR} \
    --optimizer Adamw \
    --lr_decay_step ${LR_DS} \
    --lr_decay_ratio ${LR_DR} \
    --weight_decay ${WD} \
    --unified_dim ${UD} \
    --m1_token_len 1 \
    --m2_token_len 1 \
    --use_amst True \
    --skip_factor_audio ${A_F} \
    --skip_factor_visual ${V_F} \
    --skip_factor_text ${T_F} \
    --use_helper ${USE_HELPER} \
    --helper_weight ${HELPER_WEIGHT} \
    --alpha ${ALPHA} \
    --ckpt_path ${CKPT_PATH} \
    --model_save_name ${MODEL_SAVE_NAME} \
    --fps ${FPS} \
    --train

# Optional: Use tensorboard
# Uncomment the following lines if you want to use tensorboard
# --use_tensorboard True \
# --tensorboard_path ${TENSORBOARD_PATH} \

echo "=========================================="
echo "Training Completed!"
echo "Model saved to: ${CKPT_PATH}"
echo "=========================================="
