#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list" 
NUM_GPUS=${#GPULIST[@]}
scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")

# languages=("Vietnamese" "Cambodian" "Indonesian" "Thai")
lang=$1
echo "Start Training: $lang"

CUDA_VISIBLE_DEVICES=$gpu_list torchrun --nnodes 1 --nproc_per_node $NUM_GPUS $base_dir/translation/train.py \
    --output_dir $base_dir/saved_models/llama_$lang \
    --dataset_dir $base_dir/dataset\
    --language $lang\
    --logging_strategy steps \
    --logging_steps 2 \
    --logging_first_step True \
    --save_strategy epoch \
    --save_total_limit 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --ddp_find_unused_parameters True\
    --do_eval False\
    --lr_scheduler_type cosine\
    --eval_strategy no\
    --overwrite_output_dir True\
    --remove_unused_columns True\
    --gradient_checkpointing True\
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct\
    --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct\
    --bf16 True\
    --optim adamw_bnb_8bit\
    --num_train_epochs 3\
    --trainable q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj\
    --modules_to_save lm_head,embed_tokens

# Vietnamese
# Cambodian
# Indonesian
# Thai