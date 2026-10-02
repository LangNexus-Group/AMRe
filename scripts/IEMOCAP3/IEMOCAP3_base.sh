# ============ BASE SET (Basic parameter configuration, no manual modification required) =============
# All fusion methods to be tested
# FUSIONS=("concat" "sum" "Gate" "Film" "CA" "MMTM" "CentralNet")
FUSIONS=('concat')
# Fixed hyperparameters (consistent with the original configuration)
EPOCH=60
LR_F=1e-4
LR_T=1e-4
LR_V=1e-3
LR_A=1e-3
DATASET=IEMOCAP3

ALPHA=1
AME_BETA=0.7
AME_GAMA=0.15
RANDOM_SEED=42
WARMUP_EPOCH=0
UNIFIED_DIM=768
LR_DECAY_STEP=20
WEIGHT_DECAY=1e-4
LR_DECAY_RATIO=0.1

OPTMIZER='Adamw'
MODEL_NAME='["Text","Visual","Audio"]'

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
    #     --ckpt_path "./checkpoint/$DATASET/$FUSION" \
    #     --m1_gate True \
    #     --optimizer $OPTMIZER \
    #     --lr_decay_step $LR_DECAY_STEP \
    #     --lr_decay_ratio $LR_DECAY_RATIO \
    #     --weight_decay $WEIGHT_DECAY \
        
    
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
        --model_save_name "${FUSION}_${DATASET}_with_AME" \
        --ckpt_path "./checkpoint/$DATASET/$SAVE_NAME" \
        --warmup_epoch $WARMUP_EPOCH \
        --m1_gate True \
        --optimizer $OPTMIZER \
        --lr_decay_step $LR_DECAY_STEP \
        --lr_decay_ratio $LR_DECAY_RATIO \
        --weight_decay $WEIGHT_DECAY \

done

echo -e "\n🎉 All experiments for fusion methods have been completed!"