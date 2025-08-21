import os
import sys
import logging
import structlog

# Set up structured logging
logger = structlog.get_logger()
logger.info("Testing D2R2 WF properties training")

# Ensure we're using the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comformer.train_props import train_prop_model 

# Create output directory for test
output_dir = "output/test_WF_target_space_group"
os.makedirs(output_dir, exist_ok=True)

# Print current branch to ensure we're on the right branch
import subprocess
current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
print(f"Current branch: {current_branch}")
if current_branch != "feature/training-splits":
    print("WARNING: Not on feature/training-splits branch!")

# Run a minimal test with just 1 epoch
train_prop_model(
    dataset="D2R2_surface_data",
    prop="WF",           # Use the WF field that combines WF_bottom and WF_top
    name="iComformer",   # Use invariant model for faster testing
    pyg_input=True,
    n_epochs=1,          # Just 1 epoch for testing
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,    
    use_angle=True,
    output_dir=output_dir,
    output_features=2,   # Critical: final output dimension set to 2 for WF_bottom and WF_top
    data_path="/home/mudaliar.k/data/combined_space_group.csv"  # Add explicit data path
)