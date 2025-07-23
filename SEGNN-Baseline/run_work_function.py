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

class WorkFunctionRunner(Runner):
    @property
    def default_data_path(self) -> str:
        return "Dataset/data"
    
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
        total_wf_top_mae = 0
        total_wf_top_mse = 0
        total_wf_top_mape = 0
        total_wf_bot_mae = 0
        total_wf_bot_mse = 0
        total_wf_bot_mape = 0
        num_batches = len(dataloader)

        with nullcontext() if train else torch.inference_mode():
            for data in tqdm(dataloader, f"{location} Batch", leave=False):
                data = data.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                wf_top_pred, wf_bot_pred = self.model(data)

                # Compute losses
                loss = l1_loss(wf_top_pred, data.wf_top) + l1_loss(wf_bot_pred, data.wf_bot) 

                assert not torch.isnan(loss), "Loss has hit Nan, stopping early."

                # Do training
                if train:
                    loss.backward()
                    self.optimizer.step()

                # Scale loss if normalized
                if self.config["normalize"]:
                    wf_top_pred, wf_bot_pred = self.unnormalize(wf_top_pred, wf_bot_pred)
                    wf_top_true, wf_bot_true = self.unnormalize(data.wf_top, data.wf_bot)
                else:
                    wf_top_true = data.wf_top
                    wf_bot_true = data.wf_bot

                # Deal with accuracy
                total_loss += loss.item()
                total_wf_top_mae += l1_loss(wf_top_pred, wf_top_true).item()
                total_wf_top_mse += mse_loss(wf_top_pred, wf_top_true).item()
                total_wf_top_mape += mean_absolute_percentage_error(wf_top_pred, wf_top_true).item()
                total_wf_bot_mae += l1_loss(wf_bot_pred, wf_bot_true).item()
                total_wf_bot_mse += mse_loss(wf_bot_pred, wf_bot_true).item()
                total_wf_bot_mape += mean_absolute_percentage_error(wf_bot_pred, wf_bot_true).item()
            
        return {
            f"{location}/MAE": total_loss / num_batches,
            f"{location}/WF Top MAE": total_wf_top_mae / num_batches,
            f"{location}/WF Top RMSE": (total_wf_top_mse / num_batches)**(1/2),
            f"{location}/WF Top MAPE": total_wf_top_mape / num_batches,
            f"{location}/WF Bottom MAE": total_wf_top_mae / num_batches,
            f"{location}/WF Bottom RMSE": (total_wf_top_mse / num_batches)**(1/2),
            f"{location}/WF Bottom MAPE": total_wf_top_mape / num_batches
        }
    
    def unnormalize(self, wf_top: torch.Tensor, wf_bot: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the work functions

        Args:
            wf_top: The unnormalized work function top to normalize
            wf_bot: The unnormalized work function bottom to normalize

        Returns:
            the unnormalized work function
        """
        assert wf_top.shape[-1] == 1 and wf_bot.shape[-1] == 1, "Shape of work function mismatch expected."
        coef_dict = {"wf_top": wf_top, "wf_bot": wf_bot}
        unnorm_dict = self.normalizer.unnormalize(coef_dict)
        return unnorm_dict["wf_top"], unnorm_dict["wf_bot"]
    
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
        return [Irreps("1x0e"), Irreps("1x0e")]

if __name__ == "__main__":
    runner = WorkFunctionRunner()
    runner.main()
