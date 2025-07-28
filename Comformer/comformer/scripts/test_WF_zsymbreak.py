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

logger.info("Testing WF target with z-symmetry breaking")

current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.train_props import train_prop_model 

# Create output directory first
output_dir = "output/test_WF_zsymbreak"
os.makedirs(output_dir, exist_ok=True)

# Use minimal settings for quick testing
train_prop_model(
    dataset="D2R2_surface_data",
    prop="WF",                 # Combined WF_bottom and WF_top
    name="iComformer",         # Can use either iComformer or eComformer
    pyg_input=True,
    n_epochs=5,                # Minimal epochs for testing
    max_neighbors=25,
    cutoff=4.0,
    batch_size=16,             # Smaller batch for faster iteration
    use_lattice=True,
    use_angle=True,
    break_z_symmetry=True,     # Enable z-symmetry breaking
    save_dataloader=True,
    output_dir=output_dir,
    output_features=2,         # Two outputs for WF_bottom and WF_top
    data_path="data/DFT_data.csv",  # Update with actual data path
    # Limit data for faster testing
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)