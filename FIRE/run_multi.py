from torch_geometric.loader import DataLoader
from Dataset.MultiDataset import MultiDataset
from Runner import Runner
from torch.nn.functional import l1_loss, mse_loss
import torch
from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Dict
from e3nn.o3 import Irreps

class MultiRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/Multi"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> MultiDataset:
        return MultiDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

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
        num_batches = len(dataloader)
        total_wf_top = 0 # y index 0 
        total_wf_bottom = 0 # y index 1
        total_ce = 0 # y index 2

        with nullcontext() if train else torch.inference_mode():
            for data in tqdm(dataloader, f"{location} Batch", leave=False):
                data = data.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                pred = self.model(data)[0]

                # Compute losses
                if train:
                    loss = l1_loss(pred, data.y) #l1_loss(pred, data.y) 
                else:
                    loss = l1_loss(pred, data.y)


                assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

                # Do training
                if train:
                    loss.backward()
                    self.optimizer.step()

                # Scale loss if normalized
                if self.config["normalize"]:
                    WF_pred = self.unnormalize(WF_pred)
                    WF_true = self.unnormalize(data.WF)
                    loss = l1_loss(WF_pred, WF_true) 
                else:
                    target_true = data.y

                # Deal with accuracy
                total_wf_top += l1_loss(pred[:,0], data.y[:,0]).item()
                total_wf_bottom += l1_loss(pred[:,1], data.y[:,1]).item()
                total_ce += l1_loss(pred[:,2], data.y[:,2]).item()
                total_loss += loss.item()
                total_mse += mse_loss(pred, target_true).item()
            
        return {
            f"{location}/MAE": total_loss / num_batches,
            f"{location}/RMSE": (total_mse / num_batches)**(1/2),
            f"{location}/WF Top MAE": total_wf_top / num_batches,
            f"{location}/WF Bottom MAE": total_wf_bottom / num_batches,
            f"{location}/CE MAE": total_ce / num_batches
        }
    
    def unnormalize(self, WF: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the cleavage energy

        Args:
            WF: The unnormalized WF to normalize

        Returns:
            the unnormalized WF
        """
        assert WF.shape[-1] == 2, "Shape of WF mismatch expected."
        coef_dict = {"WF": WF}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["WF"]
    
    def record_best(self, epoch: int, logs: dict):
        if self.best_logs is None:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
        elif logs["Validation/MAE"] < self.best_logs["Validation/MAE"]:
            self.best_epoch = epoch
            self.best_logs = logs
            self.best_model_state_dict = self.model.state_dict()
    
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("3x0e")]

if __name__ == "__main__":
    runner = MultiRunner()
    runner.main()
