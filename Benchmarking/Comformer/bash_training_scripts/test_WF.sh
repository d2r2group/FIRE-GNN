#!/bin/bash

#SBATCH --partition=d2r2               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=12             
#SBATCH --gres=gpu:l40s:1              
#SBATCH --mem=16G                      
#SBATCH --time=1:00:00                # Only need 1 hour for testing  
#SBATCH -o test_WF_output_%j.txt                    
#SBATCH -e test_WF_error_%j.txt                     
#SBATCH --mail-user=mudaliar.k@northeastern.edu  
#SBATCH --mail-type=ALL                     

# Debug information
echo "Current directory: $PWD"
echo "Python version and path:"
which python
python --version

module load anaconda3/2024.06
module load cuda/12.1

# Try this instead of source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate comformer_uv

cd /home/mudaliar.k/github/Comformer_uv

# Switch to the correct branch
git checkout feature/training-splits
echo "Switched to feature/training-splits branch"

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Content of data directory:"
ls -l data/

# Create output directory
mkdir -p output/test_WF_target_split

# Run the test script
python -u comformer/scripts/test_WF_target.py