# ============ BASE SET (Basic parameter configuration, no manual modification required) =============
# All fusion methods to be tested
# FUSIONS=("concat" "sum" "Gate" "Film" "CA" "MMTM" "CentralNet")
FUSIONS=("concat")
DATASET=CREMAD
EPOCH=100
LR_F=1e-3
LR_T=1e-5
LR_V=1e-3
LR_A=1e-3

ALPHA=2
LR_DE_ST=50
LR_DE_RA=0.1
AME_BETA=0
AME_GAMA=0.15
WARMUP_EPOCH=0
RANDOM_SEED=42
OPTMIZER='Adamw'
UNIFIED_DIM=512
WEIGHT_DECAY=1e-4
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
    #     --ckpt_path "./checkpoint/$DATASET/$FUSION" \
    #     --optimizer $OPTMIZER \
    #     --Use_initWeight True \
    #     --weight_decay $WEIGHT_DECAY \
    #     --lr_decay_ratio $LR_DE_RA \
    #     --lr_decay_step $LR_DE_ST \
    #     # --Use_OGM True \
        
    
    
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
        --model_save_name "${FUSION}_${DATASET}_with_AMRe" \
        --ckpt_path "./checkpoint/$DATASET/$SAVE_NAME" \
        --warmup_epoch $WARMUP_EPOCH \
        --optimizer $OPTMIZER \
        --Use_initWeight True \
        --weight_decay $WEIGHT_DECAY \
        --lr_decay_ratio $LR_DE_RA \
        --lr_decay_step $LR_DE_ST \
        --ame_gap 2 \
        --ame_gap_start 1 \
        
    
    #     # --Use_OGM True \

    # # ======================================
    # # 2. Use Shapley (MaskType=Shapley)
    # # ======================================
    # SAVE_NAME="${FUSION}_Shapley"
    # echo -e "\n=================================================="
    # echo "🔧 Starting execution: Fusion method = $FUSION | Shapley used"
    # echo "📁 Model save path: ./checkpoint/$DATASET/$SAVE_NAME"
    # echo "=================================================="
    
    # CUDA_VISIBLE_DEVICES=0 python -u ./train_all.py \
    #     --random_seed $RANDOM_SEED \
    #     --dataset $DATASET \
    #     --train \
    #     --epochs $EPOCH \
    #     --fusion_method $FUSION \
    #     --model_name $MODEL_NAME \
    #     --MaskType 'Shapley' \
    #     --alpha $ALPHA \
    #     --ame_gama $AME_GAMA \
    #     --ame_beta $AME_BETA \
    #     --unified_dim $UNIFIED_DIM \
    #     --learning_rate_fusion $LR_F \
    #     --learning_rate_audio $LR_A \
    #     --learning_rate_text $LR_T \
    #     --learning_rate_visual $LR_V \
    #     --model_save_name "${FUSION}_with_Shapley" \
    #     --ckpt_path "./checkpoint/$DATASET/$SAVE_NAME" \
    #     --warmup_epoch $WARMUP_EPOCH \
    #     --optimizer $OPTMIZER \
    #     --Use_initWeight True \
    #     # --Use_OGM True \


done

echo -e "\n🎉 All experiments for fusion methods have been completed!"