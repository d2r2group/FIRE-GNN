from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data

class MultiDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/Multi/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, forces, lattice, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        WF_top, WF_bottom = self._get_WF(idx)
        
        CE = self._get_CE(idx)
        
        # period = torch.ones_like(x); group = torch.ones_like(x)
        # torch.set_printoptions(profile="full")
        # print(x)
        # for i, elem in enumerate(x):
        #     period[i], group[i] = element_number_to_period_group(x[i].item())
        lattice = torch.broadcast_to(lattice,(len(x),9))

        return Data(x=torch.cat((x,pos[:,2].reshape(-1,1),forces),1), node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, WF_top = WF_top, WF_bottom = WF_bottom, WF=torch.cat((WF_top.reshape(-1,1),WF_bottom.reshape(-1,1)),1),
                   y=torch.cat((WF_top.reshape(-1,1),WF_bottom.reshape(-1,1),CE.reshape(-1,1)),1))
    
    def _get_WF(self, idx: int) -> torch.Tensor:

        WF_top = torch.tensor(self.molecule_data[idx]["WF_top"])
        WF_bottom = torch.tensor(self.molecule_data[idx]["WF_bottom"])
        
        WF_top = WF_top.reshape(1, 1) ; WF_bottom = WF_bottom.reshape(1, 1)
        coef_dict = {"WF_top": WF_top,
                    "WF_bottom": WF_bottom}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["WF_top"], coef_dict["WF_bottom"]
    
    def _get_CE(self, idx: int) -> torch.Tensor:

        cleavage_energy = torch.tensor(self.molecule_data[idx]["cleavage_energy"])
        cleavage_energy = cleavage_energy.reshape(1, 1)
        coef_dict = {"cleavage_energy": cleavage_energy}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["cleavage_energy"]
    