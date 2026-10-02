# ============ BASE SET (Basic parameter configuration, no manual modification required) =============
# All fusion methods to be tested
# FUSIONS=("concat" "sum" "Gate" "Film" "CA" "MMTM" "CentralNet")
FUSIONS=("concat")

EPOCH=100
LR_F=1e-3
LR_T=1e-5
LR_V=1e-3
LR_A=1e-3
DATASET=AVE

ALPHA=2
LR_DE_ST=30
LR_DE_RA=0.5
AME_BETA=0.7
AME_GAMA=0.05
RANDOM_SEED=42
WARMUP_EPOCH=5
UNIFIED_DIM=512
OPTMIZER='Adamw'
WEIGHT_DECAY=5e-4
MODEL_NAME='["Visual","Audio"]'

# ===============================================
# ========== Loop to execute experiments for all fusion methods ==========
# ===============================================
for FUSION in "${FUSIONS[@]}"; do
    # Dynamically generate the save directory name of "fusion method + AME" (e.g., concat_AME, sum_AME)
    SAVE_NAME="${FUSION}_AME"
    
    # # ======================================
    # # 1. Do not use AME (MaskType=None)
    # # ======================================
    # echo -e "\n=================================================="
    # echo "🔧 Starting execution: Fusion method = $FUSION | AME not used"
    # echo "📁 Model save path: ./checkpoint/$DATASET/$FUSION"
    # echo "=================================================="
    
    # CUDA_VISIBLE_DEVICES=0 python -u ./train_all.py \
    #     --random_seed $RANDOM_SEED \
    #     --dataset $DATASET \
    #     --train \
    #     --epochs $EPOCH \
    #     --fusion_method $FUSION \
    #     --model_name $MODEL_NAME \
    #     --MaskType 'None' \
    #     --alpha $ALPHA \
    #     --ame_gama $AME_GAMA \
    #     --ame_beta $AME_BETA \
    #     --unified_dim $UNIFIED_DIM \
    #     --learning_rate_fusion $LR_F \
    #     --learning_rate_audio $LR_A \
    #     --learning_rate_text $LR_T \
    #     --learning_rate_visual $LR_V \
    #     --model_save_name "${FUSION}_without_AME" \
    #     --ckpt_path "./checkpoint/$DATASET/${FUSION}" \
    #     --m1_gate True \
    #     --lr_decay_ratio $LR_DE_RA \
    #     --lr_decay_step $LR_DE_ST \
    #     --optimizer $OPTMIZER \
    #     --fps 2 \
    #     --weight_decay $WEIGHT_DECAY \
    #     # --LFM True \
    
    
    # # ======================================
    # 2. Use AME (MaskType=AME)
    # ======================================
    echo -e "\n=================================================="
    echo "🔧 Starting execution: Fusion method = $FUSION | AME used"
    echo "📁 Model save path: ./checkpoint/$DATASET/$SAVE_NAME"
    echo "=================================================="
    
    CUDA_VISIBLE_DEVICES=0 python -u ./train_all.py \
        --random_seed $RANDOM_SEED \
        --dataset $DATASET \
        --train \
        --epochs $EPOCH \
        --fusion_method $FUSION \
        --model_name $MODEL_NAME \
        --MaskType 'AME' \
        --alpha $ALPHA \
        --ame_gama $AME_GAMA \
        --ame_beta $AME_BETA \
        --unified_dim $UNIFIED_DIM \
        --learning_rate_fusion $LR_F \
        --learning_rate_audio $LR_A \
        --learning_rate_text $LR_T \
        --learning_rate_visual $LR_V \
        --model_save_name "${FUSION}_with_AME" \
        --ckpt_path "./checkpoint/$DATASET/${SAVE_NAME}" \
        --warmup_epoch $WARMUP_EPOCH \
        --lr_decay_ratio $LR_DE_RA \
        --lr_decay_step $LR_DE_ST \
        --optimizer $OPTMIZER \
        --fps 2 \
        --weight_decay $WEIGHT_DECAY \
        # --LFM True \

done

echo -e "\n🎉 All experiments for fusion methods have been completed!"
