#!/bin/bash
conda activate utmosv2
export CUDA_VISIBLE_DEVICES=1
export PYTHONNOUSERSITE=1
BASE_DIR="/exp/exp4/acp23xt/DTDM_Unet/Hydra_Score-Unet_TD_eval/training.train_encoder:True/dev/converted/10_steps"

# Loop through each immediate subdirectory of BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r sub_dir; do
    # Define the output file path
    out_file="$sub_dir/utmos.csv"

    # Run the inference command
    echo "Running inference on $sub_dir -> $out_file"
    python inference.py --input_dir "$sub_dir" --out_path "$out_file"
done
