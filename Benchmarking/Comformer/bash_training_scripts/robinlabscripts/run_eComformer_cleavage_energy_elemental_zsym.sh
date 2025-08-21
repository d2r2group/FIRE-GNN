#!/bin/bash

# Load GPU utility functions
source "$(dirname "$0")/gpu_utils.sh"

# Allocate GPU dynamically
ALLOCATED_GPU=$(allocate_gpu)
if [ $? -ne 0 ]; then
    echo "$ALLOCATED_GPU"
    exit 1
fi

# Set GPU environment
export_gpu_env "$ALLOCATED_GPU"

# Simulate SLURM-like job ID using timestamp
JOB_ID=$(date +%Y%m%d_%H%M%S)

# Output/error logs
OUT_FILE="eComformer_cleavage_energy_elemental_zsym_${JOB_ID}.txt"
ERR_FILE="eComformer_cleavage_energy_elemental_zsym_err_${JOB_ID}.txt"

# Start logging
{
echo "===== eComformer Cleavage Energy Elemental ZSym Training Started: $(date) ====="
echo "Hostname: $(hostname)"
echo "Working Directory: $PWD"
echo "User: $USER"
echo "Job ID: $JOB_ID"

# Debug information
echo "Current directory: $PWD"
echo "Python version and path:"
which python
python --version

# Load conda (no module system)
source /home/kmudaliar/miniconda3/etc/profile.d/conda.sh
conda activate comformer_uv

# Go to project directory (go up two levels from robinlabscripts)
cd "$(dirname "$0")/../.." || {
    echo "❌ Failed to cd into project directory"
    exit 1
}

# Switch to the correct branch
git checkout feature/training-splits
echo "✅ Switched to feature/training-splits branch"

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Content of data directory:"
ls -l ../data/

# Run the training script with Robin lab data path
export ROBINLAB_DATA_PATH="/home/kmudaliar/data"
echo "🚀 Running train_eComformer_cleavage_energy_elemental_zsym.py ..."
nohup python -u comformer/scripts/training_split_scripts/train_eComformer_cleavage_energy_elemental_zsym.py &

echo "===== eComformer Cleavage Energy Elemental ZSym Training Finished: $(date) ====="
} > "$OUT_FILE" 2> "$ERR_FILE"

echo "📊 Training completed. Check logs:"
echo "   Output: $OUT_FILE" 
echo "   Errors: $ERR_FILE"