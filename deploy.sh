#!/bin/bash
set -e
set -o pipefail
set -o xtrace


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <node_name> <model_name>"
    exit 1
fi

NODE=$1
shift
MODEL_NAME=$1

sbatch -w $NODE launch_vllm.slurm --model $MODEL_NAME --max_model_len 16000 
