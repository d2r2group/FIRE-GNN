#!/usr/bin/env python
"""
Script to generate predictions for a single property model
"""

import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm

# Add project root to path 
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from comformer.models.comformer import iComformer, iComformerConfig
from comformer.data import get_torch_test_loader

def main():
    # Define paths
    output_dir = os.path.join(root_dir, "output", "D2R2_WF_bottom")
    data_path = os.path.join(root_dir, "data", "DFT_data.csv")
    split_path = os.path.join(output_dir, "ids_train_val_test.json")
    model_path = os.path.join(output_dir, "checkpoint_350.pt")
    
    # Define the property we're predicting (same as in training script)
    target_prop = "WF_bottom"
    
    print(f"Generating predictions for {target_prop}")
    print(f"Using model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Load the train/test/val split
    with open(split_path, 'r') as f:
        train_test_val = json.loads(f.read())
        
    # Load the model
    config = iComformerConfig(
        name="iComformer",
        output_features=1,  # Single property model
        use_angle=True
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = iComformer(config).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Model loaded successfully, using device: {device}")
    
    # Create test loader
    test_loader = get_torch_test_loader(
        dataset="D2R2_surface_data",
        target=target_prop,
        batch_size=64,
        atom_features="cgcnn",
        cutoff=4.0,
        max_neighbors=25,
        id_tag="id",
        pyg_input=True,
        use_lattice=True,
        data_path=data_path,
        ids=train_test_val['id_test']
    )
    
    print(f"Test loader created with {len(test_loader.dataset)} samples")
    
    # Generate predictions
    targets = []
    predictions = []
    ids = []
    
    with torch.no_grad():
        test_ids = test_loader.dataset.ids
        for batch_idx, (dat, id) in enumerate(zip(tqdm(test_loader), test_ids)):
            g, lg, _, target = dat
            out_data = model([g.to(device), lg.to(device), _.to(device)])
            
            # Extract data
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()
            if len(target) == 1:
                target = target[0]
                
            targets.append(target)
            predictions.append(out_data)
            ids.append(id)
    
    # Save predictions to JSON file
    mem = []
    for id, target, pred in zip(ids, targets, predictions):
        info = {}
        info["id"] = id
        info["target"] = target
        info["predictions"] = pred
        mem.append(info)
    
    # Save to file
    output_file = os.path.join(output_dir, "single_prop_predictions.json")
    with open(output_file, 'w') as f:
        json.dump(mem, f, indent=2)
    
    print(f"Predictions saved to {output_file}")
    
    # Calculate metrics
    targets_np = np.array(targets)
    predictions_np = np.array(predictions)
    
    from sklearn.metrics import mean_absolute_error
    raw_mae = mean_absolute_error(targets_np, predictions_np)
    print(f"Raw MAE (standardized values): {raw_mae:.6f}")
    
if __name__ == "__main__":
    main()