#!/bin/bash


# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./configs/zero2.json

# Model configuration
llm=/your/path/to/Qwen3-VL-2B-Instruct
geo=/your/path/to/VGGT-1B # (1) VGGT-1B (2) DA3-GIANT-1.1 (3) VGGT-Omega/vggt_omega_1b_512.pt
metric=/your/path/to/DA3METRIC-LARGE
data_path=/your/data/path


# Training hyperparameters
lr=2e-5
geo_lr=1e-4
batch_size=2
grad_accum_steps=4
model_max_length=$((16*1024))

# remember to change the backend to 'decord' in transformers.video_processing_utils.py -> fetch_videos -> backend (line 820)
# I haven't find a solution to enforce decord without modifying transformers's source code; @FIXME

# data hyperparameters
datasets=vsi590k_vlm3r
max_pixels=$((576*32*32))
min_pixels=$((16*32*32))
video_max_frames=32
video_min_frames=4
video_fps=2
video_max_pixels=$(($video_max_frames*300*32*32))
video_min_pixels=$(($video_min_frames*16*32*32))

# Output configuration
run_name="VGGT-Qwen3-VL-2B-Instruct-VSI590K-VLM3R"
output_dir=/your/path/to/checkpoints/${run_name}

# Training entry point
entry_file=./training/train.py

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --geometry_encoder_path "${geo}" \
    --metric_model_path "${metric}" \
    --add_scale True \
    --add_camera True \
    --add_depth True \
    --distill_geometry_feature True \
    --data_path ${data_path} \
    --dataset_use ${datasets} \
    --camera_loss_weight 2.5 \
    --depth_loss_weight 1 \
    --distill_loss_weight 1 \
    --scale_loss_weight 1 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --min_pixels ${min_pixels} \
    --max_pixels ${max_pixels} \
    --video_min_frames ${video_min_frames} \
    --video_max_frames ${video_max_frames} \
    --video_min_pixels ${video_min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --video_fps ${video_fps} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --geo_lr ${geo_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    "

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
                    --nnodes=1 \
                    --master_port=${MASTER_PORT} \
                    ${entry_file} ${args}