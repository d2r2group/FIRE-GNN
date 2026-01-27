from typing import Dict, List
from functools import lru_cache
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from pymatgen.core import Structure
import json
from e3nn.o3 import Irreps, spherical_harmonics
import numpy as np
from torch_scatter import scatter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from p_tqdm import p_map
from functools import partial
from tqdm import tqdm

class CustomMultiDataset(Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data_file: Path, atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",   # or "float32-highest" / "float64
        )
        self.calc = ORBCalculator(orbff, device=device)
        if "cif" in molecule_data_file.name:
            parser = CifParser(molecule_data_file)
            molecule_data = parser.parse_structures()
            if type(molecule_data) is not list:
                molecule_data = [molecule_data]
        elif "json" in molecule_data_file.name:
            with molecule_data_file.open() as f:
                molecule_data = json.load(f)
            get_struct = partial(Structure.from_str, fmt="cif")
            molecule_data = p_map(get_struct, molecule_data)
        elif molecule_data_file.is_dir():
            molecule_data = p_map(Structure.from_file, molecule_data_file.glob("*.cif"))
        self.molecule_data = molecule_data
        self.forces = [self.get_forces(struc) for struc in tqdm(self.molecule_data, desc="Computing forces")]
        self.atom_init = atom_init
        self.l = Irreps.spherical_harmonics(lmax)
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.normalize = normalize

    def len(self) -> int:
        return len(self.molecule_data)

    def get_forces(self, struc: Structure):
        if not struc.is_ordered:
            for i, site in enumerate(struc):
                # Replaces the partial occupancy with just the most common species
                main_specie = max(site.species.items(), key=lambda x: x[1])[0]
                struc.replace(i, main_specie)
        atom = AseAtomsAdaptor.get_atoms(struc)
        atom.calc = self.calc
        return atom.get_forces()

    def _get(self, idx: int) -> Data:
        # Extract structure
        struc = self.molecule_data[idx]
        
        forces = self.forces[idx]
        forces = torch.tensor(forces, dtype=torch.float32)
       
        # Setup site positions
        pos = np.vstack([site.coords for site in struc])
        pos = torch.tensor(pos, dtype=torch.float32)
        # Individual atom information
        x = np.vstack([self.atom_init[str(site.specie.number)] for site in struc])
        x = torch.tensor(x, dtype=torch.float32)
        # Construct edge_index. 2xnum_edges so [0] is from and [1] is to.
        # Also constructs displacement between nodes to handle self loops
        edge_index, r_ij = self._compute_neighbors(struc, pos)
        # Edge distance to preserve that information as spherical harmonics ignore distance.
        edge_dis = torch.norm(r_ij, dim=-1, keepdim=True)
        # Construct edge attributes and node attributes from relative directions and positions
        edge_attr = spherical_harmonics(self.l, r_ij[:, [1, 2, 0]], normalize=True, normalization="component")
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean", dim_size=pos.shape[0])
        # Form graph
        return x, forces, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij
    
    def _compute_neighbors(self, struc: Structure, pos: torch.Tensor) -> torch.Tensor:
        all_nbrs = struc.get_all_neighbors(self.radius)
        # Sorts the neighbors for each site by increasing distance.
        all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance)[:self.max_neighbors] for nbrs in all_nbrs]
        source = []
        sink = []
        displacements = []
        for i, nbr in enumerate(all_nbrs):
            if len(nbr) == 0:
                # Atom is connected to nothing, so connect to everything naively
                f = [i] * pos.shape[0] + list(range(pos.shape[0]))
                t = f[::-1]
                source.extend(f)
                sink.extend(t)
                displacements.extend([pos[stop] - pos[start] for start, stop in zip(f, t)])
            else:
                # Pulls out the index of the neighbor site
                destinations = [x.index for x in nbr]
                source.extend([i] * len(destinations))
                sink.extend(destinations)
                # Compute displacement
                displacements.extend([torch.tensor(x.coords, dtype=torch.float32) - pos[i] for x in nbr])
        edge_index = np.vstack((source, sink))
        displacements = torch.vstack(displacements)
        return torch.tensor(edge_index, dtype=torch.int64), displacements

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, forces, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = self._get(idx)

        return Data(x=torch.cat((x,pos[:,2].reshape(-1,1),forces),1), node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, pos=pos, r_ij=r_ij)
    
