#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: missing arguments. Usage: $0 <config> <gpu>"
    exit 1
fi

config=$1
gpu=$2

train_set="derm7pt_train_balance"
val_set="derm7pt_val"
test_set="derm7pt_test"

repeats=3
for ((run=1; run<=repeats; run++)); do
    echo "========================================"
    echo "Running: derm, $config, run=$run, gpu=$gpu"
    echo "========================================"
    
    # launch via accelerate (DDP) when more than one GPU is selected
    IFS=',' read -ra GPUARR <<< "$gpu"
    NGPU=${#GPUARR[@]}
    if [ $NGPU -gt 1 ]; then
        PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
        python -m accelerate.commands.launch --num_processes=$NGPU --gpu_ids=$gpu --main_process_port=$PORT main.py "$train_set" "$val_set" "$config" "$test_set" -1
    else
        python main.py "$train_set" "$val_set" "$config" "$test_set" "$gpu"
    fi
done

echo "all runs finished"