from torch_geometric.loader import DataLoader
from Dataset.SlabDataset import SlabDataset
from Runner import Runner
from torch.nn.functional import l1_loss, mse_loss
from metrics import mean_absolute_percentage_error
import torch
from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Dict
from e3nn.o3 import Irreps

class CleavageEnergyRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/cleavage_energy"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> SlabDataset:
        return SlabDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, int]:
        """
        Runs an epoch on the model using the dataloader

        Args:
            dataloader: The data loader holding the data to run an epoch on
            train: Whether or not this is a training run
        
        Returns:
            The logs of the training metrics.
        """
        if train:
            self.model.train()
            location = "Train"
        else:
            self.model.eval()
            location = "Validation"

        total_loss = 0
        total_mse = 0
        total_mape = 0
        num_batches = len(dataloader)

        with nullcontext() if train else torch.inference_mode():
            for data in tqdm(dataloader, f"{location} Batch", leave=False):
                data = data.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                cleavage_energy_pred = self.model(data)[0]

                # Compute losses
                loss = l1_loss(cleavage_energy_pred, data.cleavage_energy) 

                assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

                # Do training
                if train:
                    loss.backward()
                    self.optimizer.step()

                # Scale loss if normalized
                if self.config["normalize"]:
                    cleavage_energy_pred = self.unnormalize(cleavage_energy_pred)
                    cleavage_energy_true = self.unnormalize(data.cleavage_energy)
                    loss = l1_loss(cleavage_energy_pred, cleavage_energy_true) 
                else:
                    cleavage_energy_true = data.cleavage_energy

                # Deal with accuracy
                total_loss += loss.item()
                total_mse += mse_loss(cleavage_energy_pred, cleavage_energy_true).item()
                total_mape += mean_absolute_percentage_error(cleavage_energy_pred, cleavage_energy_true).item()
            
        return {
            f"{location}/MAE": total_loss / num_batches,
            f"{location}/RMSE": (total_mse / num_batches)**(1/2),
            f"{location}/MAPE": total_mape / num_batches
        }
    
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
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            torch.save(self.model.state_dict(), self.best_model_state_dict)
        elif logs["Validation/MAE"] < self.best_logs["Validation/MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            torch.save(self.model.state_dict(), self.best_model_state_dict)
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("1x0e")]

if __name__ == "__main__":
    runner = CleavageEnergyRunner()
    runner.main()
