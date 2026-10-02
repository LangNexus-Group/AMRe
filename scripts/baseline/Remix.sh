# MASKTYPES=('None' 'AME')
# MASKTYPES=('AME')
MASKTYPES=('None')
# CREMAD 数据集 - 双模态 Remix
for MASKTYPE in "${MASKTYPES[@]}"; do
    # warm 10
    python train_Remix_New.py \
        --dataset CREMAD \
        --model_name '["Visual","Audio"]' \
        --use_remix True \
        --remix_warmup 0 \
        --modality_names "Visual,Audio" \
        --modality_gammas "2.0,2.0" \
        --batch_size 64 \
        --epochs 10 \
        --train \
        --ckpt_path ./checkpoint/Remix/CREMAD_$MASKTYPE \
        --model_save_name Remix_CREMAD_$MASKTYPE \
        --weight_decay 1e-4 \
        --fps 2 \
        --alpha 0.5 \
        --Use_initWeight True \
        --MaskType $MASKTYPE \
        --ame_gama 0.15 \
        --ame_beta 0 \

    # warm_up 5
    # AVE 数据集 - 双模态 Remix
    python train_Remix_New.py \
        --dataset AVE \
        --model_name '["Visual","Audio"]' \
        --use_remix True \
        --remix_warmup 0 \
        --modality_names "Visual,Audio" \
        --modality_gammas "2.0,2.0" \
        --batch_size 64 \
        --epochs 10 \
        --train \
        --ckpt_path ./checkpoint/Remix/AVE_$MASKTYPE  \
        --model_save_name Remix_AVE_$MASKTYPE  \
        --weight_decay 1e-4 \
        --fps 2 \
        --alpha 2 \
        --lr_decay_step 30 \
        --lr_decay_ratio 0.5 \
        --MaskType $MASKTYPE \
        --weight_decay 1e-4 \
        --ame_gama 0 \
        --ame_beta 0.7 \
        --learning_rate_audio 1e-3 \
        --learning_rate_visual 1e-3 \
        --learning_rate_fusion 1e-3 \
        
    # MVSA 数据集 - 双模态 Remix
    python train_Remix_New.py \
        --dataset MVSA \
        --model_name '["Image","Text"]' \
        --use_remix True \
        --remix_warmup 0 \
        --modality_names "Image,Text" \
        --modality_gammas "2.0,2.0" \
        --batch_size 64 \
        --epochs 10 \
        --train \
        --ckpt_path ./checkpoint/Remix/MVSA_$MASKTYPE  \
        --model_save_name Remix_MVSA_$MASKTYPE  \
        --unified_dim 768 \
        --learning_rate_image 5e-5 \
        --learning_rate_fusion 5e-5 \
        --weight_decay 0.0005 \
        --MaskType $MASKTYPE \
        --ame_gama 0 \
        --ame_beta 0.3 \
    
    # # IEMOCAP3 数据集 - 三模态 Remix
    python train_Remix_New.py \
        --dataset IEMOCAP3 \
        --model_name '["Text","Visual","Audio"]' \
        --use_remix True \
        --remix_warmup 0 \
        --modality_names "Text,Visual,Audio" \
        --modality_gammas "2.0,2.0,2.0" \
        --batch_size 64 \
        --epochs 10 \
        --train \
        --ckpt_path ./checkpoint/Remix/IEMOCAP3_$MASKTYPE  \
        --model_save_name Remix_IEMOCAP3_$MASKTYPE  \
        --unified_dim 768 \
        --learning_rate_fusion 1e-4 \
        --learning_rate_fusion 1e-4 \
        --lr_decay_step 20 \
        --lr_decay_ratio 0.5 \
        --MaskType $MASKTYPE \
        --weight_decay 5e-4 \
        --ame_gama 0.1 \
        --ame_beta 0.5 \

done
