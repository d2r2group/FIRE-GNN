import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from tqdm import tqdm
from pymatgen.core import Structure

class CrystalPointcloud_Dataset(Dataset):
    def __init__(self, csv_file, radius=7, selfedge=True):
        # Read CSV file using pandas
        data = pd.read_csv(csv_file)
        # Extract relevant columns
        self.slab_data = data['slab']
        self.cleavage_energy = data['cleavage_energy']
        self.radius = radius
        self.selfedge = selfedge

    def __len__(self):
        return len(self.slab_data)

    def __getitem__(self, idx):
        # Parse slab data to pymatgen Structure
        slab_str = self.slab_data.iloc[idx]
        slab_dict = eval(slab_str)
        slab = Structure.from_dict(slab_dict)
        
        # Extract atomic features, lattice, and coordinates
        coordinates = []
        species = []

        for site in slab:
            coordinates.append(site.coords)  # Cartesian coordinates
            species.append(site.specie.Z)    # Atomic number

        # Convert lists to tensors
        atoms = torch.tensor(species, dtype=torch.long)                   # Atomic numbers
        coords_cart = torch.tensor(coordinates, dtype=torch.float)        # Cartesian coordinates
        lattice_matrix = torch.tensor(slab.lattice.matrix, dtype=torch.float)  # Lattice vectors

        # Ensure lattice_matrix has shape [3, 3]
        if lattice_matrix.numel() == 9 and lattice_matrix.shape != (3, 3):
            lattice_matrix = lattice_matrix.view(3, 3)
        elif lattice_matrix.shape != (3, 3):
            raise ValueError(f"Unexpected lattice matrix shape: {lattice_matrix.shape}")

        # Labels: cleavage energy
        ce = torch.tensor(self.cleavage_energy.iloc[idx], dtype=torch.float).view(1, 1)

        # Create PyG Data object
        data = Data(
            x=atoms.unsqueeze(1),  # Node features: atomic numbers
            pos=coords_cart,       # Positions in Cartesian coordinates
            y=ce,                  # Cleavage energy
            atomic_numbers=atoms,
            cell=lattice_matrix,   # Lattice vectors
        )

        # Add 'natoms' attribute
        data.natoms = torch.tensor([atoms.shape[0]], dtype=torch.long)

        return data

# Save the processed dataset to a .pt file
def save_processed_dataset(dataset, save_path):
    processed_data = []
    for data in tqdm(dataset, desc=f"Processing dataset to save at {save_path}"):
        # Ensure data only contains desired attributes
        desired_keys = ['x', 'pos', 'y', 'atomic_numbers', 'cell', 'natoms']
        data = Data(**{k: v for k, v in data.items() if k in desired_keys})
        processed_data.append(data)
    torch.save(processed_data, save_path)
    print(f"Processed dataset saved to {save_path}")

# Main function to preprocess and save the dataset
def main():
    csv_file = './datasets/elemental_test_v2/WF-CE-splits.structureid-val_elements-test_0.7-0.2-0.1.test.csv'
    
    dataset = CrystalPointcloud_Dataset(csv_file, radius=7)
    # Split dataset

    save_processed_dataset(dataset, './datasets/elements_test1.pt')
   


if __name__ == "__main__":
    main()
