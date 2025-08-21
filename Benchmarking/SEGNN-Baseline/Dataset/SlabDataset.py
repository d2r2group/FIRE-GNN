from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data

class SlabDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/data/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        cleavage_energy, wf_top, wf_bot = self._get_slab_properties(idx)

        return Data(x=x, node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, cleavage_energy=cleavage_energy, wf_top=wf_top, wf_bot=wf_bot)
    
    def _get_slab_properties(self, idx: int) -> torch.Tensor:
        """
        Gets the band gap from the certain index in their normalized form

        Args:
            idx: Index of band gap to get

        Returns:
            The normalized cleavage_energy.
        """
        cleavage_energy = torch.tensor(self.molecule_data[idx]["cleavage_energy"])
        cleavage_energy = cleavage_energy.reshape(1, 1)
        wf_top = torch.tensor(self.molecule_data[idx]["WF_top"])
        wf_top = wf_top.reshape(1, 1)
        wf_bot = torch.tensor(self.molecule_data[idx]["WF_bottom"])
        wf_bot = wf_bot.reshape(1, 1)
        coef_dict = {"cleavage_energy": cleavage_energy, "wf_top": wf_top, "wf_bot": wf_bot}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["cleavage_energy"], coef_dict["wf_top"], coef_dict["wf_bot"]