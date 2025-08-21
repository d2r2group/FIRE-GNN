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

target_prop = props[2]  # Use WF_bottom as the target property
print(f"DEBUG: Target property is set to '{target_prop}'")
logger.info(f"Training model for property: {target_prop}")

# Create output directory first - simple fix to avoid directory not found error
output_dir = "output/D2R2_cleavage_energy"
os.makedirs(output_dir, exist_ok=True)

print(f"DEBUG: Calling train_prop_model with prop='{target_prop}'")
train_prop_model(
    dataset="D2R2_surface_data",
    prop=target_prop,      # Use the specified target property
    name="iComformer",
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
    data_path="/home/mudaliar.k/data/DFT_data.csv"  # Add explicit data path
)
