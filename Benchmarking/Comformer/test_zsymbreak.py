#!/usr/bin/env python
"""
Test script for verifying z-symmetry breaking implementation.
This script creates a small sample dataset and models, then verifies
that z-coordinates are properly extracted and used during forward pass.
"""

import os
import sys
import torch
import numpy as np
from pymatgen.core import Structure

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comformer.models.comformer import iComformer, iComformerConfig
from comformer.graphs import PygGraph
from jarvis.core.atoms import pmg_to_atoms

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_structure():
    """Create a test structure with distinct z-coordinates."""
    # Create a simple cubic structure with 8 atoms
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    species = ["C", "C", "C", "C", "C", "C", "C", "C"]
    
    # Create positions with distinct z-values (0, 1, 2, 3, 4)
    positions = [
        [0.0, 0.0, 0.0],  # bottom
        [2.5, 2.5, 0.0],  # bottom
        [1.0, 1.0, 1.0],  # middle-bottom
        [3.5, 3.5, 1.0],  # middle-bottom
        [1.5, 1.5, 4.0],  # top
        [3.0, 3.0, 4.0],  # top
        [0.5, 0.5, 2.0],  # middle
        [2.0, 2.0, 3.0],  # middle-top
    ]
    
    structure = Structure(lattice, species, positions)
    return structure

def check_z_coords_extraction(structure):
    """Verify z-coordinates are correctly extracted during graph creation."""
    atoms = pmg_to_atoms(structure)
    
    # Create graph
    graph = PygGraph.atom_dgl_multigraph(
        atoms=atoms,
        neighbor_strategy="k-nearest",
        cutoff=4.0,
        max_neighbors=12,
        use_canonize=True
    )
    
    # Check if z_coords is in the graph data
    if hasattr(graph, 'z_coords'):
        logger.info("✅ Z-coordinates successfully extracted")
        logger.info(f"Z-coords shape: {graph.z_coords.shape}")
        logger.info(f"Z-coords values: {graph.z_coords.squeeze().tolist()}")
        
        # Verify values match input structure
        expected_z = torch.tensor([pos[2] for pos in structure.cart_coords], dtype=torch.float).unsqueeze(1)
        match = torch.allclose(graph.z_coords, expected_z)
        logger.info(f"Z-coords match expected values: {'✅ Yes' if match else '❌ No'}")
        
        return graph
    else:
        logger.error("❌ Z-coordinates NOT found in graph data")
        return None

def check_model_zsymbreak(graph):
    """Verify model handles z-symmetry breaking correctly during forward pass."""
    # Create batch for a single graph
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph])
    batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long)
    
    # Create model configs with and without z-symmetry breaking
    config_with_zsym = iComformerConfig(
        name="iComformer",
        output_features=2,
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
    
    # Run forward pass with z-symmetry breaking enabled
    model_with_zsym.eval()
    logger.info("\nRunning forward pass with z-symmetry breaking enabled...")
    
    # Prepare input for forward pass (model expects [data, ldata, lattice] format)
    dummy_lattice = torch.zeros(1, 3, 3)  # Dummy lattice tensor
    
    with torch.no_grad():
        # Capture intermediate activations during forward pass
        orig_forward = model_with_zsym.z_embedding.forward
        
        z_inputs = []
        def hook_z_embedding(self, x):
            z_inputs.append(x.detach().clone())
            return orig_forward(x)
        
        # Apply hook
        model_with_zsym.z_embedding.forward = hook_z_embedding.__get__(
            model_with_zsym.z_embedding, model_with_zsym.z_embedding.__class__)
        
        # Run forward pass
        output_with_zsym = model_with_zsym([batch, batch, dummy_lattice])
        
        # Restore original forward
        model_with_zsym.z_embedding.forward = orig_forward
    
    # Check if z_embedding received the expected input
    if len(z_inputs) > 0:
        logger.info("✅ z_embedding network received input during forward pass")
        logger.info(f"z_embedding input shape: {z_inputs[0].shape}")
    else:
        logger.error("❌ z_embedding network did NOT receive input during forward pass")
    
    # Run forward pass without z-symmetry breaking
    model_without_zsym.eval()
    logger.info("\nRunning forward pass without z-symmetry breaking...")
    
    with torch.no_grad():
        output_without_zsym = model_without_zsym([batch, batch, dummy_lattice])
    
    # Check if outputs are different
    logger.info("\nComparing outputs:")
    logger.info(f"With z-symmetry breaking: {output_with_zsym}")
    logger.info(f"Without z-symmetry breaking: {output_without_zsym}")
    
    if not torch.allclose(output_with_zsym, output_without_zsym, atol=1e-3):
        logger.info("✅ Outputs are different - z-symmetry breaking is having an effect")
    else:
        logger.warning("❌ Outputs are identical - z-symmetry breaking might not be working properly")
    
    return output_with_zsym, output_without_zsym

def main():
    """Run the tests."""
    logger.info("=" * 50)
    logger.info("TESTING Z-SYMMETRY BREAKING IMPLEMENTATION")
    logger.info("=" * 50)
    
    # 1. Create test structure
    logger.info("\n1. Creating test structure...")
    structure = create_test_structure()
    logger.info(f"Test structure created with {len(structure)} atoms")
    logger.info(f"Z-coordinates in structure: {[pos[2] for pos in structure.cart_coords]}")
    
    # 2. Check z-coordinates extraction
    logger.info("\n2. Testing z-coordinates extraction...")
    graph = check_z_coords_extraction(structure)
    
    if graph is not None:
        # 3. Check model's handling of z-symmetry breaking
        logger.info("\n3. Testing model's handling of z-symmetry breaking...")
        outputs = check_model_zsymbreak(graph)
    
    logger.info("\n" + "=" * 50)

if __name__ == "__main__":
    main()