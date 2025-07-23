from Evaluator import Evaluator
import torch
from Dataset.SlabDataset import SlabDataset
from typing import List
from tqdm import tqdm
from e3nn.o3 import Irreps
from torch.nn.functional import l1_loss

class WorseFinder(Evaluator):
    @property
    def default_data_path(self) -> str:
        return "Dataset/elastic"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> SlabDataset:
        return SlabDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def evaluate_results(self) -> torch.Tensor:
        """
        Runs an epoch on the model using the dataloader

        Returns:
            The component-wise mae
        """
        self.model.eval()

        structures = []
        mae = []

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Test Batch", leave=False):
                data = data.to(self.device)

                cleavage_energy_pred = self.model(data)[0]

                # Scale loss if normalized
                if self.config["normalize"]:
                    cleavage_energy_pred = self.unnormalize(cleavage_energy_pred)
                    cleavage_energy_true = self.unnormalize(data.cleavage_energy)
                else:
                    cleavage_energy_true = data.cleavage_energy

                
            
        return 
    
    def unnormalize(self, cleavage_energy: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the cleavage energy

        Args:
            cleavage_energy: The unnormalized cleavage energy to normalize

        Returns:
            the unnormalized cleavage energy
        """
        assert cleavage_energy.shape[-1] == 1, "Shape of cleavage energy mismatch expected."
        coef_dict = {"cleavage_energy": cleavage_energy}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["cleavage_energy"]
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e")]

if __name__ == "__main__":
    evaluator = WorseFinder()
    evaluator.main()