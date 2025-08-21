from torch_geometric.data import Data, Dataset
import torch
from typing import Literal
import pandas as pd
from pymatgen.core.structure import Structure
import numpy as np

class SlabDataset(Dataset):
    """
    A dataset that is a collection of bulk structures and there resultant features when cut.
    """
    def __init__(self, 
                dataset: Literal["elemental_test_v2", "ptgroup_test", "spacegroup_test", "structureid_test"], 
                section: Literal["train", "validation", "test"]):
        super().__init__()
        df = pd.read_csv(f"Datasets/Data/{dataset}/{section}.csv")

        self.graphs = []

        for idx, d in df.iterrows():
            struc = Structure.from_dict(eval(d["slab"]))
            pos = np.vstack([site.coords for site in struc])
            pos = torch.tensor(pos, dtype=torch.float32)
            atoms = torch.tensor([site.specie.number for site in struc], dtype=torch.int64)
            self.graphs.append(
                Data(
                    atoms=atoms, 
                    pos=pos,
                    edge_index=self._compute_neighbors(struc, pos),
                    cleavage_energy=torch.tensor(d["cleavage_energy"], dtype=torch.float32).reshape(1,1),
                    work_function_top=torch.tensor(d["WF_top"], dtype=torch.float32).reshape(1,1),
                    work_function_bottom=torch.tensor(d["WF_bottom"], dtype=torch.float32).reshape(1,1),
                )
            )

    def _compute_neighbors(self, struc: Structure, pos: torch.Tensor) -> torch.Tensor:
        all_nbrs = struc.get_all_neighbors(8)
        # Sorts the neighbors for each site by increasing distance.
        all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance)[:12] for nbrs in all_nbrs]
        source = []
        sink = []
        for i, nbr in enumerate(all_nbrs):
            if len(nbr) == 0:
                # Atom is connected to nothing, so connect to everything naively
                f = [i] * pos.shape[0] + list(range(pos.shape[0]))
                t = f[::-1]
                source.extend(f)
                sink.extend(t)
            else:
                # Pulls out the index of the neighbor site
                destinations = [x.index for x in nbr]
                source.extend([i] * len(destinations))
                sink.extend(destinations)
        edge_index = np.vstack((source, sink))
        return torch.tensor(edge_index, dtype=torch.int64)

    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]