# # ## modality level
# CUDA_VISIBLE_DEVICES=0 python train_Resample.py \
#   --dataset CREMAD \
#   --model_name '["Visual","Audio"]' \
#   --use_resample True \
#   --resample_warmup 0 \
#   --part_ratio 0.2 \
#   --resample_alpha 1.0 \
#   --resample_func linear \
#   --batch_size 64 \
#   --epochs 10 \
#   --ckpt_path ./checkpoint/Resample/CREMAD \
#   --model_save_name cremad_resample \
#   --alpha 2 \
#   --Use_initWeight True \
#   --weight_decay 1e-4 \
#   --train

# CUDA_VISIBLE_DEVICES=0 python train_Resample.py \
#   --dataset AVE \
#   --model_name '["Visual","Audio"]' \
#   --use_resample True \
#   --resample_warmup 0 \
#   --part_ratio 0.2 \
#   --resample_alpha 1.0 \
#   --resample_func linear \
#   --batch_size 64 \
#   --epochs 10 \
#   --ckpt_path ./checkpoint/Resample/AVE \
#   --model_save_name cremad_resample \
#   --alpha 2 \
#   --lr_decay_ratio 0.5 \
#   --weight_decay 5e-4 \
#   --train \


# CUDA_VISIBLE_DEVICES=0 python train_Resample.py \
#   --dataset MVSA \
#   --model_name '["Image","Text"]' \
#   --use_resample True \
#   --resample_warmup 0 \
#   --part_ratio 0.2 \
#   --resample_alpha 1.0 \
#   --resample_func linear \
#   --batch_size 64 \
#   --epochs 10 \
#   --ckpt_path ./checkpoint/Resample/MVSA \
#   --model_save_name cremad_resample \
#   --train \
#   --unified_dim 768 \
#   --alpha 2 \
#   --learning_rate_image 5e-5 \
#   --learning_rate_fusion 5e-5 \
#   --weight_decay 5e-4 \
#   --lr_decay_ratio 0.1 \
#   --lr_decay_step 20 \

CUDA_VISIBLE_DEVICES=0 python train_Resample.py \
  --dataset IEMOCAP3 \
  --model_name '["Text","Visual","Audio"]' \
  --use_resample True \
  --resample_warmup 0 \
  --part_ratio 0.2 \
  --resample_alpha 1.0 \
  --resample_func linear \
  --batch_size 64 \
  --epochs 10 \
  --ckpt_path ./checkpoint/Resample/IEMOCAP3 \
  --model_save_name cremad_resample \
  --train \
  --unified_dim 768 \
  --learning_rate_fusion 1e-4 \
  --weight_decay 5e-4 \
  --lr_decay_ratio 0.5 \
  --lr_decay_step 20 \

# sample level

# # CREMAD数据集
# python train_Sample_level.py \
#     --dataset CREMAD \
#     --batch_size 64 \
#     --epochs 100 \
#     --sample_warmup 10 \
#     --fusion_method concat \
#     --model_name '["Visual","Audio"]' \
#     --ckpt_path ./checkpoint/Resample/sample_level/CREMAD \
#     --model_save_name cremad_sample \
#     --alpha 2 \
#     --Use_initWeight True \
#     --train

# # IEMOCAP3数据集（三模态）
# python train_Sample_level.py \
#     --dataset IEMOCAP3 \
#     --batch_size 32 \
#     --epochs 100 \
#     --sample_warmup 5 \
#     --use_video_frames 3 \
#     --fusion_method concat \
#     --model_name '["Text","Visual","Audio"]' \
#     --ckpt_path ./checkpoint/Resample/sample_level/IEMOCAP3 \
#     --model_save_name iemocap3_sample \
#     --unified_dim 768 \
#     --train