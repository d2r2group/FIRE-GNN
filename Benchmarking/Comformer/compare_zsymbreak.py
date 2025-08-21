#!/usr/bin/env python
"""
Comparison script for z-symmetry breaking feature.
This script loads a small sample of real data and compares:
1. Model predictions with and without z-symmetry breaking
2. Feature importance analysis for z-coordinates
3. Visual inspection of learned z-coordinate embeddings
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comformer.models.comformer import iComformer, iComformerConfig, eComformer, eComformerConfig
from comformer.graphs import PygGraph
from comformer.data import load_dataset, get_pyg_dataset
from jarvis.core.atoms import pmg_to_atoms
from torch_geometric.data import Batch

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_small_dataset(data_path="data/DFT_data.csv", n_samples=5):
    """Load a small portion of the dataset for testing."""
    logger.info(f"Loading {n_samples} samples from {data_path}")
    df = load_dataset(
        name="D2R2_surface_data", 
        data_path=data_path,
        target="WF",
        limit=n_samples
    )
    logger.info(f"Loaded {len(df)} samples")
    return df

def create_graph_data(df):
    """Create graph data from the dataframe."""
    # Get pyg dataset
    dataset, mean, std = get_pyg_dataset(
        dataset=df.to_dict('records'),
        id_tag="jid",
        target="WF",
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        use_canonize=True,
        name="D2R2_surface_data",
        line_graph=True,
        cutoff=4.0,
        max_neighbors=25,
        use_lattice=True,
        use_angle=True
    )
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    logger.info(f"Mean: {mean}, Std: {std}")
    
    return dataset, mean, std

def create_models():
    """Create models with and without z-symmetry breaking."""
    # Create model configs
    config_with_zsym = iComformerConfig(
        name="iComformer",
        output_features=2,  # WF_bottom and WF_top
        break_z_symmetry=True
    )
    
    config_without_zsym = iComformerConfig(
        name="iComformer",
        output_features=2,
        break_z_symmetry=False
    )
    
    # Create models
    model_with_zsym = iComformer(config_with_zsym)
    model_without_zsym = iComformer(config_without_zsym)
    
    return model_with_zsym, model_without_zsym

def compare_predictions(dataset, model_with_zsym, model_without_zsym):
    """Compare predictions with and without z-symmetry breaking."""
    logger.info("Comparing predictions...")
    
    # Set models to eval mode
    model_with_zsym.eval()
    model_without_zsym.eval()
    
    results = []
    
    with torch.no_grad():
        for i, (g, lg, lattice, y) in enumerate(dataset[:5]):
            # Prepare a batch with a single graph
            batch = Batch.from_data_list([g])
            batch_lg = Batch.from_data_list([lg]) 
            batch_lattice = lattice.unsqueeze(0) if len(lattice.shape) < 3 else lattice
            
            # Get predictions
            pred_with_zsym = model_with_zsym([batch, batch_lg, batch_lattice])
            pred_without_zsym = model_without_zsym([batch, batch_lg, batch_lattice])
            
            # Convert to numpy for easier manipulation
            y_np = y.numpy()
            pred_with_zsym_np = pred_with_zsym.numpy()
            pred_without_zsym_np = pred_without_zsym.numpy()
            
            # Calculate squared error
            se_with_zsym = ((pred_with_zsym_np - y_np) ** 2).mean()
            se_without_zsym = ((pred_without_zsym_np - y_np) ** 2).mean()
            
            # Check if z_coords exists
            has_z_coords = hasattr(g, 'z_coords')
            z_range = g.z_coords.max() - g.z_coords.min() if has_z_coords else None
            
            results.append({
                'sample_idx': i,
                'true_values': y_np.tolist(),
                'pred_with_zsym': pred_with_zsym_np.tolist(),
                'pred_without_zsym': pred_without_zsym_np.tolist(),
                'se_with_zsym': se_with_zsym,
                'se_without_zsym': se_without_zsym,
                'diff_percentage': (abs(se_with_zsym - se_without_zsym) / max(se_with_zsym, se_without_zsym)) * 100,
                'has_z_coords': has_z_coords,
                'z_range': z_range
            })
            
            logger.info(f"Sample {i}:")
            logger.info(f"  True values: {y_np}")
            logger.info(f"  With z-sym: {pred_with_zsym_np}")
            logger.info(f"  Without z-sym: {pred_without_zsym_np}")
            logger.info(f"  Squared error with z-sym: {se_with_zsym:.4f}")
            logger.info(f"  Squared error without z-sym: {se_without_zsym:.4f}")
            logger.info(f"  Difference: {((abs(se_with_zsym - se_without_zsym) / max(se_with_zsym, se_without_zsym)) * 100):.2f}%")
    
    # Analyze results
    df_results = pd.DataFrame(results)
    logger.info("\nSummary of results:")
    logger.info(f"Average squared error with z-sym: {df_results['se_with_zsym'].mean():.4f}")
    logger.info(f"Average squared error without z-sym: {df_results['se_without_zsym'].mean():.4f}")
    
    better_count = sum(df_results['se_with_zsym'] < df_results['se_without_zsym'])
    logger.info(f"Z-symmetry breaking improved predictions in {better_count}/{len(df_results)} samples")
    
    # Find correlation between z-coordinate range and improvement
    if 'z_range' in df_results and not df_results['z_range'].isna().all():
        corr = df_results['z_range'].corr(df_results['se_without_zsym'] - df_results['se_with_zsym'])
        logger.info(f"Correlation between z-range and error reduction: {corr:.4f}")
        logger.info("A positive correlation indicates that z-symmetry breaking helps more when the z-range is larger")
    
    return df_results

def visualize_z_embedding(model_with_zsym, dataset):
    """Visualize z-coordinate embeddings for a few samples."""
    logger.info("Visualizing z-coordinate embeddings...")
    
    # Set model to eval mode
    model_with_zsym.eval()
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (g, lg, lattice, y) in enumerate(dataset[:3]):
            if i >= 3:  # Limit to 3 samples for clearer visualization
                break
                
            if not hasattr(g, 'z_coords'):
                logger.warning(f"Sample {i} does not have z_coords attribute")
                continue
            
            # Get z-coordinates
            z_coords = g.z_coords.numpy()
            
            # Get z-embeddings directly
            z_embeddings = model_with_zsym.z_embedding(g.z_coords).numpy()
            
            # Create a subplot
            plt.subplot(3, 1, i+1)
            
            # Plot each z-coordinate's embedding as a heatmap
            plt.imshow(z_embeddings, aspect='auto', cmap='viridis')
            plt.colorbar(label='Embedding Value')
            
            # Add z-coordinate values as y-axis labels
            plt.yticks(range(len(z_coords)), [f"{z[0]:.2f}" for z in z_coords])
            
            plt.title(f"Sample {i} - Z-Coordinate Embeddings")
            plt.xlabel('Embedding Dimension')
            plt.ylabel('Z-Coordinate Value')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('z_embedding_visualization.png')
    logger.info("Saved z-embedding visualization to z_embedding_visualization.png")

def main():
    """Run the comparison tests."""
    logger.info("=" * 50)
    logger.info("COMPARING Z-SYMMETRY BREAKING IMPLEMENTATION")
    logger.info("=" * 50)
    
    # 1. Load a small dataset
    logger.info("\n1. Loading small dataset...")
    try:
        df = load_small_dataset()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Please adjust the data_path in load_small_dataset() to point to your actual data file.")
        return
    
    # 2. Create graph data
    logger.info("\n2. Creating graph data...")
    try:
        dataset, mean, std = create_graph_data(df)
    except Exception as e:
        logger.error(f"Error creating graph data: {e}")
        return
    
    # 3. Create models
    logger.info("\n3. Creating models...")
    model_with_zsym, model_without_zsym = create_models()
    
    # 4. Compare predictions
    logger.info("\n4. Comparing predictions...")
    try:
        results = compare_predictions(dataset, model_with_zsym, model_without_zsym)
    except Exception as e:
        logger.error(f"Error comparing predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Visualize z-embeddings
    logger.info("\n5. Visualizing z-embeddings...")
    try:
        visualize_z_embedding(model_with_zsym, dataset)
    except Exception as e:
        logger.error(f"Error visualizing z-embeddings: {e}")
        return
    
    logger.info("\n" + "=" * 50)
    logger.info("Test complete!")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()