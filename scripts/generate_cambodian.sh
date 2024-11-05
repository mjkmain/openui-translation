#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list" 
NUM_GPUS=${#GPULIST[@]}
scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")

lang="Cambodian"

model_name="mjkmain/llama3.1-cambodian"
echo "Inference Model: $model_name"

for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 $base_dir/translation/evaluation/generate_response.py \
    --model_name_or_path "$model_name" \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target_language $lang\
    --gpu_id $gpu_id \
    --raw_data_path mjkmain/translation-full\
    --dataset_dir $base_dir/translation/dataset\
    --num_gpus $NUM_GPUS &
done