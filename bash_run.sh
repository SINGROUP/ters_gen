#!/bin/bash

for config in configs/*.yaml; do 
echo "Running with config: $config"
sbatch train_parameter_search.sh $config
done