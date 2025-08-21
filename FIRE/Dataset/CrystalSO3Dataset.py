import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from pymatgen.core.structure import Structure
from typing import Dict, List, Optional
from e3nn.o3 import Irreps, spherical_harmonics
from torch_scatter import scatter
from abc import ABCMeta, abstractmethod
from .Normalization import Normalizer
import json

class CrystalSO3Dataset(Dataset, metaclass=ABCMeta):
    """
    A dataset that provides data about a crystal including
    a representation of its positioning using SO(3) spherical
    harmonics.
    """

    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8.0, max_neighbors: int = 12, normalize: bool = True, coef_path: Optional[str] = None):
        """
        Initializes a crystal dataset from a list of molecule data

        Args:
            molecule_data: A list of dictionaries that are of form 
                            {"material": material_dict, **{other tensor data and decompositions}}
            atom_init: A dictionary of atom element number (z) to their embedding tensor.
            lmax: The maximum l required with spherical harmonics
            radius: The furthest distance from an atom to look for edge connections in Angstroms.
            max_neighbors: How many neighbors to include for each atom in the crystal graph.
            normalize: Whether or not to normalize the target 
            coef_path: Where the stats about the spherical harmonic decomposition can be found.
        """
        super().__init__()
        self.molecule_data = molecule_data
        self.atom_init = atom_init
        self.l = Irreps.spherical_harmonics(lmax)
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.normalize = normalize
        with open(coef_path) as f:
            self.normalizer = Normalizer(json.load(f), "cpu")

    def len(self) -> int:
        return len(self.molecule_data)

    @abstractmethod
    def get(self, idx: int) -> Data:
        # Extract structure
        struc = Structure.from_dict(self.molecule_data[idx]["material"])
        lattice = torch.tensor(struc.lattice.matrix.flatten(), dtype=torch.float32)
        
        forces = torch.tensor(self.molecule_data[idx]["forces"])
       
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
        return x, forces, lattice, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij
    
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