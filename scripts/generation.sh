#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list" 
NUM_GPUS=${#GPULIST[@]}
# echo "Active GPUs : ${gpu_list}"

scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")


###########################################################################
model_name="/home/maverick/openui-translation/saved_models/llama_Vietnamese"
language="Vietnamese"
echo "Inference Model : $model_name"

for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 $base_dir/translation/evaluation/generate_response.py \
    --model_name_or_path "$model_name" \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --target_language $language\
    --gpu_id $gpu_id \
    --num_gpus $NUM_GPUS &
done
wait
###########################################################################

# ###########################################################################
# model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
# language="Cambodian"
# echo "Inference Model : $model_name"

# for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 $base_dir/translation/evaluation/generate_response.py \
#     --model_name_or_path "$model_name" \
#     --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --target_language $language\
#     --gpu_id $gpu_id \
#     --num_gpus $NUM_GPUS &
# done
# wait
# ###########################################################################

# ###########################################################################
# model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
# language="Indonesian"
# echo "Inference Model : $model_name"

# for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 $base_dir/translation/evaluation/generate_response.py \
#     --model_name_or_path "$model_name" \
#     --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --target_language $language\
#     --gpu_id $gpu_id \
#     --num_gpus $NUM_GPUS &
# done
# wait
# ###########################################################################

# ###########################################################################
# model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
# language="Thai"
# echo "Inference Model : $model_name"

# for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 $base_dir/translation/evaluation/generate_response.py \
#     --model_name_or_path "$model_name" \
#     --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --target_language $language\
#     --gpu_id $gpu_id \
#     --num_gpus $NUM_GPUS &
# done
# wait
###########################################################################

# Vietnamese
# Cambodian
# Indonesian
# Thai