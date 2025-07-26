import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from tqdm import tqdm
from pymatgen.core import Structure
from normalization import compute_target_stats
class CrystalPointcloud_Dataset(Dataset):
    def __init__(self, csv_file, mu, std, radius=8.0, selfedge=True):
        # Read CSV file using pandas
        data = pd.read_csv(csv_file)
        # Extract relevant columns
        self.slab_data = data['slab']
        # For work function training, extract WF_bottom and WF_top
        self.wf_bottom = data['WF_bottom']
        self.wf_top = data['WF_top']
        self.mu  = torch.tensor(mu,  dtype=torch.float32)   # shape (2,)
        self.std = torch.tensor(std, dtype=torch.float32)
        # If you still need cleavage_energy, you can store it too:
        # self.cleavage_energy = data['cleavage_energy']
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

        # (Optional) Convert list of coordinates to a numpy array first
        # to speed up tensor creation:
        import numpy as np
        coords_cart = torch.tensor(np.array(coordinates), dtype=torch.float)

        # Convert species to tensor
        atoms = torch.tensor(species, dtype=torch.long)
        lattice_matrix = torch.tensor(slab.lattice.matrix, dtype=torch.float)
        if lattice_matrix.numel() == 9 and lattice_matrix.shape != (3, 3):
            lattice_matrix = lattice_matrix.view(3, 3)
        elif lattice_matrix.shape != (3, 3):
            raise ValueError(f"Unexpected lattice matrix shape: {lattice_matrix.shape}")

        # Compute work function target as a 2-element tensor
        wf_bottom_val = self.wf_bottom.iloc[idx]
        wf_top_val = self.wf_top.iloc[idx]
        # ce_val = self.cleavage_energy.iloc[idx]
        wf = torch.tensor([wf_bottom_val, wf_top_val], dtype=torch.float)
        wf_norm = (wf - self.mu) / self.std
        new_x = torch.cat([atoms.unsqueeze(1).float(), coords_cart[:, 2].unsqueeze(1)], dim=1)
        # new_x = atoms.unsqueeze(1).float()
        # Create PyG Data object with the work function target
        data = Data(
            x=new_x,
            pos=coords_cart,
            y=wf_norm,
            atomic_numbers=atoms,
            cell=lattice_matrix,
        )
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
    csv_file = './datasets/results_20230524_439654hrs_final.csv'
    try:
        stats = np.load("./datasets/wf_norm_stats.npz")
        mu, std = stats["mu"], stats["std"]
    except FileNotFoundError:
        mu, std = compute_target_stats(csv_file)
    dataset = CrystalPointcloud_Dataset(csv_file, mu, std, radius=8.0)
    # Save the processed datasets
    for idx in tqdm(range(len(dataset)), desc="Checking slabs"):
        data = dataset[idx]
        z = data.pos[:,2].numpy()
        z_sorted = np.sort(z)

        # find the largest gap (vacuum) in the sorted z’s
        diffs = np.diff(z_sorted)
        gap_idx = int(np.argmax(diffs))

        bottom_max = z_sorted[gap_idx]
        top_min    = z_sorted[gap_idx + 1]

        # flag if bottom slab reaches above the split plane
        if bottom_max > top_min:
            print(f"⚠️ Sample {idx}: bottom_z_max = {bottom_max:.4f} > top_z_min = {top_min:.4f}")
    # save_processed_dataset(dataset, './datasets/wfstructureid_test.pt')


if __name__ == "__main__":
    main()

