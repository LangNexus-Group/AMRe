# ============ BASE SET (Basic parameter configuration, no manual modification required) =============
# All fusion methods to be tested
# FUSIONS=("concat" "sum" "Gate" "Film" "CA" "MMTM" "CentralNet")
FUSIONS=("concat")
FUSION="concat"
LR_F=1e-3
LR_T=1e-3
LR_V=1e-3
LR_A=1e-3


# # Data set 
# AVE
EPOCH=100
DATASET=AVE
MODEL_NAME='["Visual","Audio"]'
ALPHA=2
LR_DE_ST=30
LR_DE_RA=0.5
AME_BETA=0.7
AME_GAMA=0.15
AME_GAP=3


UNIFIED_DIM=3584
WEIGHT_DECAY=5e-4
WARMUP_EPOCH=10
RANDOM_SEED=42
OPTMIZER='Adamw'

METRIC="ce_loss"
BATCH_SIZE=64
for FUSION in "${FUSIONS[@]}"; do
    
    CUDA_VISIBLE_DEVICES=0 python -u ./train_all_OLA.py \
        --random_seed $RANDOM_SEED \
        --dataset $DATASET \
        --train \
        --epochs $EPOCH \
        --fusion_method $FUSION \
        --model_name $MODEL_NAME \
        --MaskType 'None' \
        --alpha $ALPHA \
        --ame_gama $AME_GAMA \
        --ame_beta $AME_BETA \
        --unified_dim $UNIFIED_DIM \
        --learning_rate_fusion $LR_F \
        --learning_rate_audio $LR_A \
        --learning_rate_text $LR_T \
        --learning_rate_visual $LR_V \
        --model_save_name "${FUSION}_${DATASET}_with_AMRe" \
        --ckpt_path "./checkpoint/$DATASET/Concat/base" \
        --warmup_epoch $WARMUP_EPOCH \
        --optimizer $OPTMIZER \
        --weight_decay $WEIGHT_DECAY \
        --lr_decay_ratio $LR_DE_RA \
        --lr_decay_step $LR_DE_ST \
        --ame_gap $AME_GAP \
        --Use_OLA True \

    SAVE_NAME="${FUSION}_AME"
    CUDA_VISIBLE_DEVICES=0 python -u ./train_all_OLA.py \
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
        --ckpt_path "./checkpoint/$DATASET/${SAVE_NAME}_${METRIC}/OLA" \
        --warmup_epoch $WARMUP_EPOCH \
        --optimizer $OPTMIZER \
        --weight_decay $WEIGHT_DECAY \
        --lr_decay_ratio $LR_DE_RA \
        --lr_decay_step $LR_DE_ST \
        --ame_gap $AME_GAP \
        --ame_gap_start 0 \
        --ame_acc_metric $METRIC \
        --ame_unc_metric 'kl' \
        --Use_OLA True \
        --Use_MACE True \

done