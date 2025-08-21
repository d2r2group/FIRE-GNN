import os
import sys
import logging

# Set up regular logging as fallback if structlog isn't available
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger.info("structlog not available, using standard logging")

logger.info("Starting D2R2 WF training with z-symmetry breaking")

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

# Create output directory first
output_dir = "output/D2R2_WF_zsymbreak"
os.makedirs(output_dir, exist_ok=True)

train_prop_model(
    dataset="D2R2_surface_data",
    prop="WF",                 # Combined WF_bottom and WF_top 
    name="eComformer",         # Use equivariant model for better performance
    pyg_input=True,
    n_epochs=350,
    max_neighbors=25,
    cutoff=4.0,
    batch_size=64,
    use_lattice=True,          # Required for proper graph construction
    use_angle=True,
    break_z_symmetry=True,     # Enable z-symmetry breaking
    save_dataloader=True,
    output_dir=output_dir,
    output_features=2,         # Two outputs for WF_bottom and WF_top
    data_path="/home/mudaliar.k/data/DFT_data.csv",  # Add explicit data path  # Update with actual data path
)