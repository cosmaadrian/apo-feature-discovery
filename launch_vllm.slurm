#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# Default values
NUM_GPUS="1"
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
PORT="3000"
MAX_MODEL_LEN="16000"
VLLM_VERSION="v0.11.0"  # Default version
REQUIREMENTS_FILE="requirements.txt"
VENV_PATH="$HOME/venvs/vllm_venv"
GPU_MEM_UTIL="auto"  # New: 'auto' for dynamic calculation, or a fixed value like "0.9"

export HF_TOKEN=$(cat ./.hf_token)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --version) VLLM_VERSION="$2"; shift 2 ;;
    --requirements) REQUIREMENTS_FILE="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --sif) SIF_PATH="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;  # New argument
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# Build the image name
mkdir -p "${HOME}/sif_images"
DOCKER_IMAGE="docker://vllm/vllm-openai:${VLLM_VERSION}"
SIF_PATH="${HOME}/sif_images/vllm_${VLLM_VERSION}.sif"

# Build SIF if not exists
if [ ! -f "$SIF_PATH" ]; then
    echo "SIF image not found. Building from Docker..."
    
    singularity build --fakeroot "${SIF_PATH}" "${DOCKER_IMAGE}"
    singularity cache clean -f
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build SIF image"
        exit 1
    fi
    
    echo "SIF image built successfully: ${SIF_PATH}"
else
    echo "Using existing SIF image: ${SIF_PATH}"
fi

DOCKER_VERSION="v$(singularity exec --nv ${SIF_PATH} vllm --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
VENV_PATH="${VENV_PATH}_${DOCKER_VERSION}"

calculate_gpu_memory_utilization() {
    # Get GPU memory info (in MiB)
    GPU_TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    GPU_USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    
    # Stampa info su stderr così non interferisce col return value
    echo "" >&2
    echo "=========================================" >&2
    echo "Calculating optimal GPU memory utilization..." >&2
    echo "=========================================" >&2
    echo "GPU Total Memory: ${GPU_TOTAL_MEM} MiB" >&2
    echo "GPU Used Memory:  ${GPU_USED_MEM} MiB" >&2
    echo "GPU Free Memory:  ${GPU_FREE_MEM} MiB" >&2
    
    # Convert to GiB for readability
    GPU_TOTAL_GIB=$(awk "BEGIN {printf \"%.2f\", $GPU_TOTAL_MEM / 1024}")
    GPU_FREE_GIB=$(awk "BEGIN {printf \"%.2f\", $GPU_FREE_MEM / 1024}")
    GPU_USED_GIB=$(awk "BEGIN {printf \"%.2f\", $GPU_USED_MEM / 1024}")
    
    echo "GPU Total Memory: ${GPU_TOTAL_GIB} GiB" >&2
    echo "GPU Used Memory:  ${GPU_USED_GIB} GiB" >&2
    echo "GPU Free Memory:  ${GPU_FREE_GIB} GiB" >&2
    
    # Calculate utilization ratio (free / total)
    # Leave a 5% safety margin
    CALCULATED_RATIO=$(awk "BEGIN {printf \"%.4f\", ($GPU_FREE_MEM / $GPU_TOTAL_MEM) * 0.95}")
    
    # Clamp between 0.3 and 0.95
    # If calculated ratio is too low (< 0.3), warn and use 0.3
    # If calculated ratio is > 0.95, cap at 0.95
    IS_TOO_LOW=$(awk "BEGIN {print ($CALCULATED_RATIO < 0.3)}")
    IS_TOO_HIGH=$(awk "BEGIN {print ($CALCULATED_RATIO > 0.95)}")
    
    if [ "$IS_TOO_LOW" = "1" ]; then
        echo "" >&2
        echo "WARNING: Calculated ratio is very low ($CALCULATED_RATIO)" >&2
        echo "This means GPU is heavily occupied. Setting to minimum 0.3" >&2
        echo "Consider killing other processes or waiting for GPU to free up." >&2
        echo "" >&2
        CALCULATED_RATIO="0.3"
    elif [ "$IS_TOO_HIGH" = "1" ]; then
        CALCULATED_RATIO="0.95"
    fi
    
    # Round to 2 decimal places for cleaner output
    FINAL_RATIO=$(awk "BEGIN {printf \"%.2f\", $CALCULATED_RATIO}")
    
    echo "" >&2
    echo "Calculated GPU Memory Utilization: ${FINAL_RATIO}" >&2
    echo "(This reserves ~5% margin for CUDA operations)" >&2
    echo "=========================================" >&2
    echo "" >&2
    
    echo "$FINAL_RATIO"
}

# Determine GPU memory utilization
if [ "$GPU_MEM_UTIL" = "auto" ]; then
    GPU_MEM_UTIL=$(calculate_gpu_memory_utilization)
else
    echo "Using manually specified GPU memory utilization: ${GPU_MEM_UTIL}"
fi

echo "GPU MEMORY UTIL VALUE: ${GPU_MEM_UTIL}"


# Log configuration
echo ""
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPUs: $NUM_GPUS"
echo "vLLM Version: $VLLM_VERSION"
echo "Requirements: $REQUIREMENTS_FILE"
echo "Venv: $VENV_PATH"
echo "Using Docker image: $SIF_PATH"
echo "========================================="
echo "========================================="
echo ""
echo "SSH tunnel command:"
echo "ssh -L ${PORT}:${SLURM_NODELIST}:${PORT} ${USER}@thor.supsi.ch"
echo ""

HOME_CACHE=$HOME/.cache
HOME_CACHE=$(realpath "$HOME_CACHE")

echo "Resolved path for ${USER} cache dir is: ${HOME_CACHE}"
echo ""

mkdir -p "${HOME_CACHE}/huggingface"
mkdir -p "${HOME_CACHE}/flashinfer"



DISABLE_FLASHINFER=""
case "$VLLM_VERSION" in
    v0.7.*|v0.8.*|v0.9.*|v0.10.*|v0.11.*|latest)
        DISABLE_FLASHINFER="--env VLLM_USE_FLASHINFER_SAMPLER=0"
        echo "FlashInfer sampler disabled (recommended for V100)"
        ;;
    *)
        echo "Using default FlashInfer sampler settings"
        ;;
esac

# Setup venv and install requirements, if not already present
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating venv at $VENV_PATH..."
    singularity exec --nv $SIF_PATH python3 -m venv $VENV_PATH
    
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing requirements..."
        singularity exec --nv \
            --bind $VENV_PATH:$VENV_PATH \
            $SIF_PATH \
            $VENV_PATH/bin/pip install -r $REQUIREMENTS_FILE
    fi
else
    echo "Venv already exists at $VENV_PATH"
fi

echo "Venv ready at $VENV_PATH"


singularity exec --nv \
    --no-home \
    --bind "${HOME_CACHE}:/home/${USER}/.cache" \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="/home/${USER}/.cache/huggingface" \
    --env FLASHINFER_WORKSPACE_DIR="/home/${USER}/.cache/flashinfer" \
    $DISABLE_FLASHINFER \
    $SIF_PATH \
    vllm serve "${MODEL_PATH}" \
        --host "0.0.0.0" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --tensor-parallel-size "${NUM_GPUS}" \
        --dtype float16 \
        --gpu-memory-utilization "${GPU_MEM_UTIL}"

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
