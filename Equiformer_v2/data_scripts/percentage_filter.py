import torch
import numpy as np

def filter_pt_file(input_file, output_file, percentile=95):
    # Load the processed dataset (assumed to be a list of PyG Data objects)
    dataset = torch.load(input_file)
    print(f"Loaded {len(dataset)} graphs from {input_file}")

    # Extract the number of nodes (atoms) for each graph
    natoms_list = [data.natoms.item() for data in dataset]
    
    # Compute the threshold at the given percentile (default: 95th percentile)
    threshold = np.percentile(natoms_list, percentile)
    print(f"Filtering graphs with more than {threshold} nodes (>{percentile}th percentile)")

    # Filter out graphs that have a number of nodes above the threshold
    filtered_data = [data for data in dataset if data.natoms.item() <= threshold]
    num_filtered = len(dataset) - len(filtered_data)
    print(f"Filtered out {num_filtered} graphs out of {len(dataset)}")

    # Save the filtered dataset
    torch.save(filtered_data, output_file)
    print(f"Filtered dataset saved to {output_file}\n")

if __name__ == '__main__':
    # Define your file paths
    train_in = './datasets/sgnum_train.pt'
    val_in = './datasets/sgnum_validation.pt'
    test_in = './datasets/sgnum_test.pt'
    
    train_out = './datasets/sgnum_filtered_train.pt'
    val_out = './datasets/sgnum_filtered_validation.pt'
    test_out = './datasets/sgnum_filtered_test.pt'
    
    # Process each file
    filter_pt_file(train_in, train_out, percentile=95)
    filter_pt_file(val_in, val_out, percentile=95)
    filter_pt_file(test_in, test_out, percentile=99)

