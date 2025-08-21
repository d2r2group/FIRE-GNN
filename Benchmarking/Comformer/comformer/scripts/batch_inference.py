#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch inference script for Comformer models trained on surface properties.
This script loads a trained model and applies it to predict properties 
(WF_TOP, WF_BOTTOM, Cleavage energy) for structures in a CSV file.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to import comformer modules
current_dir = os.path.dirname(os.path.abspath(__file__))
comformer_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(comformer_dir)

from comformer.models.comformer import iComformer, iComformerConfig
from comformer.data import load_dataset, load_pyg_graphs
from comformer.graphs import PygStructureDataset
from torch.utils.data import DataLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch inference with trained Comformer model")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the trained model checkpoint")
    parser.add_argument("--input_csv", type=str, required=True,
                      help="Path to input CSV file with structures")
    parser.add_argument("--output_csv", type=str, required=True,
                      help="Path to output CSV file with predictions")
    parser.add_argument("--cutoff", type=float, default=4.0,
                      help="Cutoff distance for graph construction")
    parser.add_argument("--max_neighbors", type=int, default=25,
                      help="Maximum number of neighbors for graph construction")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for inference")
    
    return parser.parse_args()


def load_model(model_path):
    """Load the trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    # Configure model with 3 output features for multi-property prediction
    config = iComformerConfig(name="iComformer", output_features=3)
    model = iComformer(config)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    model.eval()  # Set model to evaluation mode
    
    print(f"Loaded model from {model_path}")
    return model, checkpoint.get("mean_train", 0), checkpoint.get("std_train", 1)


def prepare_dataset(input_csv, cutoff, max_neighbors, mean_train, std_train):
    """Prepare dataset for inference."""
    # Load data
    df = load_dataset(data_path=input_csv, target="all")
    
    # Create PyG graphs for the structures
    graphs = load_pyg_graphs(
        df,
        neighbor_strategy="k-nearest",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=True,
        use_angle=True
    )
    
    # Create dataset
    dataset = PygStructureDataset(
        df,
        graphs,
        target="all",
        atom_features="cgcnn",
        line_graph=False,
        id_tag="jid",
        classification=False,
        neighbor_strategy="k-nearest",
        mean_train=mean_train,
        std_train=std_train,
    )
    
    return dataset, df


def run_inference(model, dataset, batch_size, device="cpu"):
    """Run inference on the dataset."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        drop_last=False,
    )
    
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_ids = [data["id"] for data in batch[3]]
            batch_data = (batch[0].to(device), batch[1], batch[2])
            
            outputs = model(batch_data)
            
            # For multi-property prediction, outputs will have shape [batch_size, 3]
            # If outputs is squeezed to 1D tensor for a single sample, reshape it
            if len(outputs.shape) == 1 and len(batch_ids) == 1:
                outputs = outputs.unsqueeze(0)
                
            preds = outputs.cpu().numpy()
            predictions.extend(preds)
            ids.extend(batch_ids)
    
    return ids, predictions


def save_predictions(df, ids, predictions, output_csv, mean_train, std_train):
    """Save predictions to a CSV file."""
    # Create a new DataFrame with predictions
    results_df = pd.DataFrame({
        "jid": ids,
        "WF_bottom_pred": [],
        "WF_top_pred": [],
        "cleavage_energy_pred": []
    })
    
    # Unstandardize predictions if mean and std are provided
    preds_array = np.array(predictions)
    if mean_train is not None and std_train is not None:
        # Check if mean_train and std_train are lists/arrays for multi-output
        if isinstance(mean_train, (list, np.ndarray)) and len(mean_train) == 3:
            preds_array = preds_array * np.array(std_train) + np.array(mean_train)
        else:
            preds_array = preds_array * std_train + mean_train
    
    # Add predictions to results DataFrame
    results_df["WF_bottom_pred"] = preds_array[:, 0]
    results_df["WF_top_pred"] = preds_array[:, 1]
    results_df["cleavage_energy_pred"] = preds_array[:, 2]
    
    # Merge with original DataFrame to include original properties
    merged_df = pd.merge(df, results_df, on="jid", how="left")
    
    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Device setup - use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, mean_train, std_train = load_model(args.model_path)
    model.to(device)
    
    # Prepare dataset
    dataset, df = prepare_dataset(
        args.input_csv, 
        args.cutoff, 
        args.max_neighbors,
        mean_train,
        std_train
    )
    
    # Run inference
    ids, predictions = run_inference(model, dataset, args.batch_size, device)
    
    # Save predictions
    save_predictions(df, ids, predictions, args.output_csv, mean_train, std_train)


if __name__ == "__main__":
    main()