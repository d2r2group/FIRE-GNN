from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data

class WFDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/WF/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        WF_top, WF_bottom = self._get_WF(idx)

        return Data(x=torch.cat((x,pos[:,2].reshape(-1,1)),1), node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, WF_top = WF_top, WF_bottom = WF_bottom, WF=torch.cat((WF_top.reshape(-1,1),WF_bottom.reshape(-1,1)),1))
    
    def _get_WF(self, idx: int) -> torch.Tensor:
        """
        Gets the band gap from the certain index in their normalized form

        Args:
            idx: Index of band gap to get

        Returns:
            The normalized WF.
        """
        WF_top = torch.tensor(self.molecule_data[idx]["WF_top"])
        WF_bottom = torch.tensor(self.molecule_data[idx]["WF_bottom"])
        
        WF_top = WF_top.reshape(1, 1) ; WF_bottom = WF_bottom.reshape(1, 1)
        coef_dict = {"WF_top": WF_top,
                    "WF_bottom": WF_bottom}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["WF_top"], coef_dict["WF_bottom"]