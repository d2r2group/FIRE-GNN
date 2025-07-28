import os
import sys
import logging
import structlog

# Set up structured logging
logger = structlog.get_logger()
logger.info("Starting D2R2 single property training")

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

props = [
    "WF_bottom",
    "WF_top",
    "cleavage_energy",
]

target_prop = props[1]  # Use WF_top as the target property
print(f"DEBUG: Target property is set to '{target_prop}'")
logger.info(f"Training model for property: {target_prop}")

# Include date and time as a unique identifier in the output directory name
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output/D2R2_WF_top_augmented_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Also copy the data file to the output directory for reference
data_path = "/home/mudaliar.k/data/DFT_data_augmented.csv"  # The data file we'll be using
import shutil
try:
    shutil.copy2(data_path, os.path.join(output_dir, os.path.basename(data_path)))
    logger.info(f"Copied {data_path} to {output_dir}")
except Exception as e:
    logger.error(f"Failed to copy data file: {e}")

print(f"DEBUG: Calling train_prop_model with prop='{target_prop}'")
train_prop_model(
    dataset="D2R2_surface_data",
    prop=target_prop,      # Use the specified target property
    name="eComformer",
    pyg_input=True,
    n_epochs=400,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,
    use_angle=True,
    save_dataloader=True,
    output_dir=output_dir,
    output_features=1,    # Output dimension set to 1 for single property
    data_path=data_path   # Using the augmented dataset defined above
)
