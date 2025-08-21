#!/bin/bash

#SBATCH --partition=d2r2               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=12             
#SBATCH --gres=gpu:l40s:1              
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH -o iComformer_cleavage_energy_elemental_%j.txt                    
#SBATCH -e iComformer_cleavage_energy_elemental_err_%j.txt                     
#SBATCH --mail-user=mudaliar.k@northeastern.edu  
#SBATCH --mail-type=ALL                     

module load anaconda3/2024.06
module load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate comformer_uv
cd /home/mudaliar.k/github/Comformer_uv
git checkout feature/training-splits

python -u comformer/scripts/training_split_scripts/train_iComformer_cleavage_energy_elemental.py