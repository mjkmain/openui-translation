#!/bin/bash
scripts=$(readlink -f "$0")
scripts_dir=$(dirname "$scripts")
base_dir=$(dirname "$scripts_dir")

languages=("Vietnamese" "Cambodian" "Indonesian" "Thai")

        # --result_path /home/maverick/openui-translation/tuning_results \
for lang in "${languages[@]}"; do
    echo $lang
    python3 "$base_dir/translation/evaluation/scoring.py" \
        --lang $lang \
        --result_path "$base_dir/scripts/results/" \
        --tokenizer_name_or_path google-bert/bert-base-multilingual-cased
    wait
done
# Vietnamese
# Cambodian
# Indonesian
# Thai