# ============ BASE SET (Basic parameter configuration, no manual modification required) =============
# All fusion methods to be tested
# FUSIONS=("concat" "sum" "Gate" "Film" "CA" "MMTM" "CentralNet")
FUSIONS=("concat")
# Fixed hyperparameters (consistent with the original configuration)
EPOCH=60
LR_F=5e-5
LR_T=1e-5
LR_V=5e-5
LR_A=1e-3
DATASET=MVSA
ALPHA=2
AME_GAMA=0.1
AME_BETA=0.7
UNIFIED_DIM=768
RANDOM_SEED=42
WARMUP_EPOCH=0
MODEL_NAME='["Image","Text"]'
OPTMIZER='Adamw'
WEIGHT_DECAY=5e-4
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
    #     --learning_rate_image $LR_V \
    #     --model_save_name "${FUSION}_without_AME" \
    #     --ckpt_path "./checkpoint/$DATASET/${FUSION}" \
    #     --m1_gate True \
    #     --optimizer $OPTMIZER \
    #     --weight_decay $WEIGHT_DECAY \
    #     --lr_decay_step 20 \
    #     --lr_decay_ratio 0.5 \
    #     # --LFM True
    
    # ======================================
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
        --learning_rate_image $LR_V \
        --warmup_epoch $WARMUP_EPOCH \
        --m1_gate True \
        --optimizer $OPTMIZER \
        --weight_decay $WEIGHT_DECAY \
        --lr_decay_step 30 \
        --lr_decay_ratio 0.5 \
        --model_save_name "${FUSION}_${DATASET}_with_AME" \
        --ckpt_path "./checkpoint/$DATASET/${SAVE_NAME}" \
        # --LFM True \
        # --model_save_name "${FUSION}_with_AME" \
        # --ckpt_path "./checkpoint/$DATASET/${SAVE_NAME}" \
        

done