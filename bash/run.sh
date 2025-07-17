#!/bin/bash

CONFIG_DIR="/home/sethih1/masque_new/composnet/ters_gen/configs_hypopt"
TRAIN_SCRIPT="/home/sethih1/masque_new/composnet/ters_gen/train_parameter_search.sh"

# loop through all yaml files
for config_path in "$CONFIG_DIR"/*.yaml; do
    # extract the filename without path and extension
    config_file=$(basename "$config_path")
    job_name="${config_file%.*}"

    echo "Submitting job: $job_name with config: $config_path"
    sbatch --job-name="$job_name" "$TRAIN_SCRIPT" "$config_path"
done
